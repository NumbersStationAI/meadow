from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from meadow.database.connector.connector import Column, Connector, Table


class DuckDBConnector(Connector):
    """Connector for DuckDB."""

    def __init__(self, db_path: str) -> None:
        """Create DuckDB connector."""
        self.db_path = db_path
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database file {self.db_path} does not exist.")
        self.conn: duckdb.DuckDBPyConnection = None
        self.cur: duckdb.DuckDBPyConnection = None

    @property
    def dialect(self) -> str:
        """Get the dialect of the database."""
        return "duckdb"

    def quote(self, value: str) -> str:
        return f'"{value}"'

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = duckdb.connect(self.db_path)
        self.cur = self.conn.cursor()

    def close(self) -> None:
        """Close the connection to the database."""
        if self.conn:
            self.conn.close()

    def commit(self) -> None:
        """Commit changes to the database."""
        if self.cur:
            self.cur.commit()

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        if not self.cur:
            raise ValueError("Must connect to database before running SQL.")
        return self.cur.sql(sql).df()

    def execute_sql(self, sql: str, parameters: Any = None) -> None:
        """Run the SQL without returning anything."""
        if not self.cur:
            raise ValueError("Must connect to database before running SQL.")
        self.cur.execute(sql, parameters=parameters)
        self.commit()

    def get_tables(self) -> list[Table]:
        """Get the tables in the database."""
        # read from information_schema.tables
        sql = """
SELECT table_name, column_name, data_type
FROM information_schema.columns;
"""
        df = self.run_sql_to_df(sql)
        tables = []
        for table_name in df["table_name"].unique():
            table_df = df[df["table_name"] == table_name]
            columns: list[Column] = []
            for _, row in table_df.iterrows():
                columns.append(
                    Column(
                        name=row["column_name"],
                        data_type=row["data_type"],
                    )
                )
            # Try to get data deterministically
            column_str = ", ".join([c.name for c in columns])
            data_sample_query = (
                f"SELECT DISTINCT * FROM {table_name} ORDER BY {column_str} LIMIT 5"
            )
            data_df = self.run_sql_to_df(data_sample_query)
            tables.append(
                Table(
                    name=table_name,
                    columns=columns,
                    data=data_df.to_dict(orient="records"),
                )
            )
        return tables
