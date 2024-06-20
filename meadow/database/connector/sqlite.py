"""SQLite connector for Meadow."""

import sqlite3
from pathlib import Path
from typing import Any

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

    def quote(self, value: str) -> str:
        return f"'{value}'"

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.text_factory = lambda b: b.decode(errors="ignore")
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

    def execute_sql(self, sql: str, parameters: Any = None) -> None:
        """Run the SQL without returning anything."""
        self.conn.execute(sql, parameters=parameters)  # type: ignore
        self.commit()

    def get_column_sample_values(
        self, table_name: str, column_name: str, limit: int = 3
    ) -> list:
        """Get sample values for a column."""
        sql = f"""
SELECT DISTINCT {column_name}
FROM {table_name}
LIMIT {limit}
ORDER BY RANDOM();
"""
        return self.run_sql_to_df(sql)[column_name].tolist()

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
            columns = []
            for row in columns_df.itertuples():
                # Get sample values for the column
                # sample_values = self.get_column_sample_values(table_name, row.column_name)  # type: ignore
                columns.append(Column(name=row.column_name, data_type=row.data_type))  # type: ignore
            columns = [
                Column(name=row.column_name, data_type=row.data_type)  # type: ignore
                for row in columns_df.itertuples()
            ]
            # Try to get data deterministically
            column_str = ", ".join([c.name for c in columns])
            data_sample_query = (
                f"SELECT DISTINCT * FROM {table_name} ORDER BY {column_str} LIMIT 5"
            )
            df = self.run_sql_to_df(data_sample_query)
            tables.append(
                Table(
                    name=table_name, columns=columns, data=df.to_dict(orient="records")
                )
            )
        return tables
