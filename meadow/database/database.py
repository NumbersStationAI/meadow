import logging

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
        data_type = df[column_name].dtype.name
        sql_dtype = map_dtype_to_sql(data_type)
        columns.append(Column(name=column_name, data_type=sql_dtype))
    return columns


# TODO: does this need to be a class or can it just be a pydantic thing
class Database:
    """Representation of database."""

    def __init__(self, connector: Connector) -> None:
        """Initialize the database."""
        self._connector = connector

        self._connector.connect()
        self._base_tables = {tbl.name: tbl for tbl in self._connector.get_tables()}
        self._connector.close()

        self._view_tables: dict[str, Table] = {}

    @property
    def tables(self) -> list[Table]:
        """Get the tables in the database."""
        return list(self._base_tables.values()) + list(self._view_tables.values())

    def get_table(self, name: str) -> Table | None:
        """Get the table by name."""
        if name in self._base_tables:
            return self._base_tables[name]
        if name in self._view_tables:
            return self._view_tables[name]
        return None

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        sql = self.normalize_query(sql)
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
        try:
            df = self.run_sql_to_df(sql)
            estimated_columns = extract_columns_from_df(df)
        except Exception as e:
            logger.error(f"Failed to extract columns from view sql. sql={sql}, e={e}")
        if not override and name in self._view_tables:
            raise ValueError(f"View already exists with name {name}")
        self._view_tables[name] = Table(
            name=name,
            description=description,
            is_view=True,
            columns=estimated_columns,
            view_sql=sql,
        )

    def remove_view(self, name: str) -> None:
        """Remove a view from the database."""
        if name in self._view_tables:
            del self._view_tables[name]

    def get_number_of_views(self) -> int:
        """Return the number of views in the database."""
        return len(self._view_tables)

    def normalize_query(self, sql: str) -> str:
        """Return SQL over base tables by replacing any subqueries from history."""
        view_sql_pairs = [
            (name, table.view_sql) for name, table in self._view_tables.items()
        ]
        return add_views_as_ctes(sql, view_sql_pairs)
