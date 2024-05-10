import logging

import pandas as pd
import sqlglot

from meadow.database.connector.connector import Connector, Table

logger = logging.getLogger(__name__)


def validate_sql(sql: str, dialect: str) -> tuple[bool, str | None]:
    """Validate the SQL."""
    try:
        sqlglot.parse_one(sql, dialect=dialect)
        return True, None
    except Exception as e:
        logger.error(f"Failed to parse SQL. sql={sql}, e={e}")
        return False, str(e)


def replace_view_select_star_with_def(
    sql: str, view_sql_pairs: list[tuple[str, str]]
) -> str:
    """Replace SELECT * FROM <view_name> with the view definition."""
    for view_name, view_sql in view_sql_pairs:
        if view_name not in sql:
            continue
        if f"(SELECT * FROM {view_name})" in sql:
            sql = sql.replace(
                f"(SELECT * FROM {view_name})", f"({view_sql}) AS {view_name}"
            )
        else:
            sql = sql.replace(f"{view_name}", f"({view_sql}) AS {view_name}")
    return sql


# TODO: does this need to be a class or can it just be a pydantic thing
class Database:
    """Representation of database."""

    def __init__(self, connector: Connector) -> None:
        """Initialize the database."""
        self._connector = connector

        self._connector.connect()
        self._base_tables = self._connector.get_tables()
        self._connector.close()

        self._view_tables: list[Table] = []

    @property
    def tables(self) -> list[Table]:
        """Get the tables in the database."""
        return self._base_tables + self._view_tables

    def get_table(self, name: str) -> Table | None:
        """Get the table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        self._connector.connect()
        result = self._connector.run_sql_to_df(sql)
        self._connector.close()
        return result

    def add_view(self, name: str, sql: str) -> None:
        """Add a view to the database."""
        valid, error = validate_sql(sql, self._connector.dialect)
        if not valid:
            raise ValueError(
                f"Invalid SQL sql={sql}, dialect={self._connector.dialect}, error={error}"
            )
        self._view_tables.append(Table(name=name, is_view=True, view_sql=sql))

    def remove_view(self, name: str) -> None:
        """Remove a view from the database."""
        self._view_tables = [table for table in self._view_tables if table.name != name]

    def normalize_query(self, sql: str) -> str:
        """Return SQL over base tables by replacing any subqueries from history."""
        view_sql_pairs = [(table.name, table.view_sql) for table in self._view_tables]
        return replace_view_select_star_with_def(sql, view_sql_pairs)
