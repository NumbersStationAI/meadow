import logging
from functools import partial

import pandas as pd
import sqlglot
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes

from meadow.database.connector.connector import Column, Connector, Table

logger = logging.getLogger(__name__)


def validate_sql(sql: str, dialect: str) -> tuple[bool, str | None]:
    """Validate the SQL."""
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True, None
    except Exception as e:
        logger.error(f"Failed to parse SQL. sql={sql}, e={e}")
        return False, str(e)


def _remap_table_sql(
    node: sqlglot.Expression, remapping: dict[str, str]
) -> sqlglot.Expression:
    """Remap tables."""
    if isinstance(node, sqlglot.exp.Column):
        if node.table in remapping:
            node.args["table"] = sqlglot.exp.Identifier(
                this=remapping[node.table], quoted=False
            )
    elif isinstance(node, sqlglot.exp.Table):
        if node.this.this in remapping:
            node.this.args["this"] = remapping[node.this.this]
    return node


def _get_base_remapping_sql(base_table: Table, new_table: Table) -> str:
    """Get the SQL for remapping base tables."""
    column_aliases = []
    for old_column, new_column in zip(base_table.columns, new_table.columns):
        column_aliases.append(f"{old_column.name} AS {new_column.name}")
    sql = f"SELECT {', '.join(column_aliases)} "
    sql += f"FROM {base_table.name}"
    return sql


def add_views_as_ctes(
    sql: str, view_sql_pairs: list[tuple[str, str]], dialect: str = "sqlite"
) -> str:
    """ADD <view_name>, <view_def> as CTES.."""
    if not view_sql_pairs:
        return sql
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    # Get existing with statements to add _after_ the rest of the views are added
    # Otherwise eliminate_ctes could remove required views because they assume
    # the CTE order is if view A is required in view B, then view A is first.
    existing_ctes = None
    if "with" in parsed.args and parsed.args["with"]:
        existing_ctes = parsed.args["with"]
        del parsed.args["with"]
    for view_name, view_sql in view_sql_pairs:
        parsed = parsed.with_(view_name, view_sql)
    if existing_ctes:
        parsed.args["with"].args["expressions"].extend(
            existing_ctes.args["expressions"]
        )
    parsed = eliminate_ctes(parsed)
    return parsed.sql(dialect=dialect, pretty=True)


def add_base_table_remaps_as_ctes(
    sql: str,
    base_table_remapping: dict[str, Table],
    base_tables: dict[str, Table],
    dialect: str = "sqlite",
) -> str:
    """Add base table remaps as CTEs."""
    if not base_table_remapping:
        return sql

    base_table_view_sql_pairs = []
    for k, table in base_table_remapping.items():
        base_table_view_sql_pairs.append(
            (f"{k}_", _get_base_remapping_sql(base_tables[k], table))
        )

    parsed = sqlglot.parse_one(sql, dialect=dialect)
    # First, we need to change the table names of the remapped tables globally
    # so when we create base table CTEs, there is no name collision
    base_table_name_mapping = {k: f"{k}_" for k in base_table_remapping.keys()}
    parsed = parsed.transform(
        partial(_remap_table_sql, remapping=base_table_name_mapping)
    )

    existing_ctes = None
    if "with" in parsed.args and parsed.args["with"]:
        existing_ctes = parsed.args["with"]
        del parsed.args["with"]
    for view_name, view_sql in base_table_view_sql_pairs:
        parsed = parsed.with_(view_name, view_sql)
    if existing_ctes:
        parsed.args["with"].args["expressions"].extend(
            existing_ctes.args["expressions"]
        )
    try:
        parsed = eliminate_ctes(parsed)
    except Exception as e:
        logger.warning(f"Failed to eliminate CTEs. sql={sql}, e={e}")
        pass

    return parsed.sql(dialect=dialect, pretty=True)


def map_dtype_to_sql(dtype: str) -> str:
    """Map pandas data type to SQLite data type."""
    mapping = {
        "int64": "INTEGER",
        "float64": "REAL",
        "object": "TEXT",
        "bool": "INTEGER",
        "datetime64[ns]": "TEXT",  # SQLite does not have a dedicated datetime type
        "timedelta[ns]": "TEXT",
    }
    if dtype not in mapping:
        import ipdb

        ipdb.set_trace()
        logger.warning(f"Unknown dtype {dtype}")
    return mapping.get(dtype, "TEXT")  # Default to TEXT if dtype is not found


def extract_columns_from_df(df: pd.DataFrame) -> list[Column]:
    """Extract columns from the dataframe."""
    columns = []
    for column_name in df.columns:
        if len(df[column_name].shape) == 2 and df[column_name].shape[1] > 1:
            data_type = df[column_name].iloc[:, 0].dtype.name
        else:
            data_type = df[column_name].dtype.name
        sql_dtype = map_dtype_to_sql(data_type)
        columns.append(Column(name=column_name, data_type=sql_dtype))
    return columns


def check_if_non_select_query(sql: str) -> bool:
    """Check if the SQL is a non-select query."""
    parsed = sqlglot.parse_one(sql, dialect="sqlite")
    if (
        parsed.find(sqlglot.exp.Create)
        or parsed.find(sqlglot.exp.Drop)
        or parsed.find(sqlglot.exp.Update)
        or parsed.find(sqlglot.exp.Insert)
        or parsed.find(sqlglot.exp.Delete)
        or parsed.find(sqlglot.exp.AlterTable)
        or parsed.find(sqlglot.exp.AlterColumn)
    ):
        return True
    return False


class Database:
    """Representation of database."""

    def __init__(self, connector: Connector) -> None:
        """Initialize the database."""
        self._connector = connector

        self._connector.connect()
        self._base_tables: dict[str, Table] = {
            tbl.name: tbl for tbl in self._connector.get_tables()
        }
        self._connector.close()
        self._base_table_remapping: dict[str, Table] = {}

        self._view_tables: dict[str, Table] = {}

    @property
    def tables(self) -> list[Table]:
        """Get the tables in the database."""
        base_tables = []
        for k in self._base_tables:
            if k in self._base_table_remapping:
                if not self._base_table_remapping[k].is_deprecated:
                    base_tables.append(self._base_table_remapping[k])
            else:
                if not self._base_tables[k].is_deprecated:
                    base_tables.append(self._base_tables[k])
        return base_tables + list(self._view_tables.values())

    def get_table(self, name: str) -> Table | None:
        """Get the table by name."""
        if (
            name in self._base_table_remapping
            and not self._base_table_remapping[name].is_deprecated
        ):
            return self._base_table_remapping[name]
        if name in self._base_tables and not self._base_tables[name].is_deprecated:
            return self._base_tables[name]
        if name in self._view_tables:
            return self._view_tables[name]
        return None

    def finalize_draft_views(self) -> None:
        """Finalize all draft views."""
        for _, table in self._view_tables.items():
            if table.is_draft:
                table.is_draft = False

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        sql = self.normalize_query(sql)
        if check_if_non_select_query(sql):
            raise ValueError("Only SELECT queries are allowed.")
        self._connector.connect()
        result = self._connector.run_sql_to_df(sql)
        self._connector.close()
        return result

    def add_view(
        self, name: str, sql: str, description: str = None, override: bool = True
    ) -> None:
        """Add a view to the database."""
        valid, error = validate_sql(sql, self._connector.dialect)
        if not valid:
            raise ValueError(
                f"Invalid SQL sql={sql}, dialect={self._connector.dialect}, error={error}"
            )
        # Try to mine for columns from the sql
        estimated_columns = None
        df = None
        try:
            df = self.run_sql_to_df(sql)
        except Exception as e:
            logger.error(f"Failed to run view sql. sql={sql}, e={e}")
            raise e
        try:
            estimated_columns = extract_columns_from_df(df)
        except Exception as e:
            logger.error(f"Failed to extract columns from view sql. sql={sql}, e={e}")
            raise e
        if not override and name in self._view_tables:
            raise ValueError(f"View already exists with name {name}")
        self._view_tables[name] = Table(
            name=name,
            description=description,
            is_view=True,
            is_draft=True,
            columns=estimated_columns,
            data=df.head(5).to_dict(orient="records") if df is not None else None,
            view_sql=sql,
        )

    def add_base_table_column_remap(
        self, name: str, column_remap: dict[str, str]
    ) -> None:
        """Add a base table column name remapping.

        This will deprecate the old base table and add a new one that is treated as a view
        in SQL compilation.
        """
        base_table = self._base_tables.get(name)
        if base_table is None:
            return None
        new_table = base_table.model_copy(deep=True)
        for old_column_name, new_column_name in column_remap.items():
            for column in new_table.columns:
                if column.name == old_column_name:
                    column.name = new_column_name
                    break
        # Rename the data, too, but keep column order
        for i, row in enumerate(new_table.data):
            new_table.data[i] = {column_remap.get(k, k): v for k, v in row.items()}
        self._base_table_remapping[name] = new_table
        self._base_tables[name].is_deprecated = True
        return

    def remove_view(self, name: str) -> None:
        """Remove a view from the database."""
        if name in self._view_tables:
            del self._view_tables[name]

    def get_number_of_views(self) -> int:
        """Return the number of views in the database."""
        return len(self._view_tables)

    def normalize_query(self, sql: str) -> str:
        """Return SQL over base tables with CTEs from views and base table remappings."""
        view_sql_pairs = [
            (name, table.view_sql) for name, table in self._view_tables.items()
        ]
        viewed_sql = add_views_as_ctes(sql, view_sql_pairs)
        # Add base table views for the column remaps
        base_table_sql = add_base_table_remaps_as_ctes(
            viewed_sql, self._base_table_remapping, self._base_tables
        )
        return base_table_sql
