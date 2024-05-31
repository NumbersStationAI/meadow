from pathlib import Path
from typing import Generator

import duckdb
import pandas as pd
import pytest

from meadow.database.connector.connector import Table
from meadow.database.connector.duckdb import DuckDBConnector


@pytest.fixture
def duckdb_connector(tmp_path: Path) -> Generator[DuckDBConnector, None, None]:
    """Fixture to setup and teardown a DuckDB database connection."""
    # Create a temporary DuckDB file and populate it
    db_path = tmp_path / "test.db"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    conn.close()

    # Yield a DuckDBConnector instance for testing
    connector = DuckDBConnector(db_path=str(db_path))
    yield connector

    # Remove the DuckDB file after the test run
    db_path.unlink()


def test_duckdb_invalid_db_path() -> None:
    """Test initialization of DuckDBConnector with invalid db_path."""
    with pytest.raises(FileNotFoundError) as e:
        DuckDBConnector(db_path="invalid_path")
    assert "Database file invalid_path does not exist." in str(e.value)


def test_duckdb_connector_init(duckdb_connector: DuckDBConnector) -> None:
    """Test initialization of DuckDBConnector."""
    assert duckdb_connector.db_path.endswith(
        "test.db"
    ), "Database file should be set correctly."


def test_duckdb_quote(duckdb_connector: DuckDBConnector) -> None:
    """Test quoting values for DuckDB."""
    assert duckdb_connector.quote("test") == '"test"', "Value should be quoted."


def test_duckdb_connector_connect(duckdb_connector: DuckDBConnector) -> None:
    """Test connection to DuckDB."""
    duckdb_connector.connect()
    assert duckdb_connector.conn is not None, "Connection should be established."


def test_duckdb_connector_get_tables(duckdb_connector: DuckDBConnector) -> None:
    """Test retrieving tables and their schema."""
    duckdb_connector.connect()
    tables = duckdb_connector.get_tables()
    assert isinstance(tables, list), "Should return a list of tables."
    assert len(tables) == 1, "Should find one table."
    table = tables[0]
    assert isinstance(table, Table), "Should contain Table objects."
    assert table.name == "test_table", "Table name should match."
    assert isinstance(table.columns, list), "Columns should be a list."
    assert len(table.columns) == 2, "There should be two columns."
    assert table.columns[0].name == "id", "First column should be 'id'."
    assert table.columns[1].name == "name", "Second column should be 'name'."
    assert table.data == [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]


@pytest.mark.parametrize(
    "sql, expected_len",
    [("SELECT * FROM test_table WHERE id = 1;", 1), ("SELECT * FROM test_table;", 2)],
)
def test_parametrized_sql_queries(
    duckdb_connector: DuckDBConnector, sql: str, expected_len: int
) -> None:
    """Test various SQL queries."""
    duckdb_connector.connect()
    df = duckdb_connector.run_sql_to_df(sql)
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame."
    assert len(df) == expected_len, "DataFrame should contain two rows."
    assert list(df.columns) == ["id", "name"], "DataFrame should have correct columns."
