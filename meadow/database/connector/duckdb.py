from pathlib import Path

import duckdb
import pandas as pd

from meadow.database.connector.connector import Column, Connector, Table


class DuckDBConnector(Connector):
    """Connector for DuckDB."""

    def __init__(self, cache_file: str) -> None:
        """Create DuckDB connector."""
        self.cache_file = cache_file
        if not Path(self.cache_file).exists():
            raise FileNotFoundError(f"Cache file {self.cache_file} does not exist.")
        self.conn: duckdb.DuckDBPyConnection = None

    @property
    def dialect(self) -> str:
        """Get the dialect of the database."""
        return "duckdb"

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = duckdb.connect(self.cache_file)

    def commit(self) -> None:
        """Commit changes to the database."""
        self.conn.commit()

    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        return self.conn.sql(sql).df()

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
            columns = []
            for _, row in table_df.iterrows():
                columns.append(
                    Column(
                        name=row["column_name"],
                        data_type=row["data_type"],
                    )
                )
            tables.append(Table(name=table_name, columns=columns))
        return tables
