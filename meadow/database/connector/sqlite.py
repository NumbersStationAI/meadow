"""SQLite connector for Meadow."""

import sqlite3
from pathlib import Path

import pandas as pd

from meadow.database.connector.connector import Column, Connector, Table


class SQLiteConnector(Connector):
    """Connector for SQLite."""

    def __init__(self, db_path: str) -> None:
        """Create SQLite connector."""
        self.db_path = db_path
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database file {self.db_path} does not exist.")
        self.conn: sqlite3.Connection = None

    @property
    def dialect(self) -> str:
        """Get the dialect of the database."""
        return "sqlite"

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enables column access by name

    def close(self) -> None:
        """Close the connection to the database."""
        if self.conn:
            self.conn.close()

    def commit(self) -> None:
        """Commit changes to the database."""
        if self.conn:
            self.conn.commit()

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        return pd.read_sql_query(sql, self.conn)

    def get_tables(self) -> list[Table]:
        """Get the tables in the database."""
        sql = """
SELECT tbl_name as table_name
FROM sqlite_master
WHERE type='table';
"""
        tables_df = self.run_sql_to_df(sql)
        tables = []
        for table_name in tables_df["table_name"].unique():
            column_sql = f"""
SELECT name as column_name, type as data_type
FROM pragma_table_info('{table_name}');
"""
            columns_df = self.run_sql_to_df(column_sql)
            columns = [
                Column(name=row.column_name, data_type=row.data_type)
                for row in columns_df.itertuples()
            ]
            tables.append(Table(name=table_name, columns=columns))
        return tables
