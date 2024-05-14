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


def add_views_as_ctes(
    sql: str, view_sql_pairs: list[tuple[str, str]], dialect: str = "sqlite"
) -> str:
    """ADD <view_name>, <view_def> as CTES.."""

    views_to_add = []
    for view_name, view_sql in view_sql_pairs:
        if view_name in sql:
            views_to_add.append((view_name, view_sql))
    if not views_to_add:
        return sql
    parsed = sqlglot.parse_one(sql, dialect=dialect)
    for view_name, view_sql in views_to_add:
        parsed = parsed.with_(view_name, view_sql)
    return parsed.sql(dialect=dialect)


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
        self._view_tables[name] = Table(name=name, is_view=True, view_sql=sql)

    def remove_view(self, name: str) -> None:
        """Remove a view from the database."""
        if name in self._view_tables:
            del self._view_tables[name]

    def normalize_query(self, sql: str) -> str:
        """Return SQL over base tables by replacing any subqueries from history."""
        view_sql_pairs = [
            (name, table.view_sql) for name, table in self._view_tables.items()
        ]
        return add_views_as_ctes(sql, view_sql_pairs)
