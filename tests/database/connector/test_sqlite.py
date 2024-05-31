import sqlite3
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from meadow.database.connector.connector import Table
from meadow.database.connector.sqlite import SQLiteConnector


@pytest.fixture
def sqlite_connector(tmp_path: Path) -> Generator[SQLiteConnector, None, None]:
    """Fixture to setup and teardown a SQLiteDB database connection."""
    # Create a temporary SQLiteDB file and populate it
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    conn.close()

    # Yield a SQLiteConnector instance for testing
    connector = SQLiteConnector(db_path=str(db_path))
    yield connector

    # Remove the SQLiteDB file after the test run
    db_path.unlink()


def test_sqlite_invalid_db_path() -> None:
    """Test initialization of SQLiteConnector with invalid db_path."""
    with pytest.raises(FileNotFoundError) as e:
        SQLiteConnector(db_path="invalid_path")
    assert "Database file invalid_path does not exist." in str(e.value)


def test_sqlite_connector_init(sqlite_connector: SQLiteConnector) -> None:
    """Test initialization of SQLiteConnector."""
    assert sqlite_connector.db_path.endswith(
        "test.db"
    ), "Database file should be set correctly."


def test_sqlite_quote(sqlite_connector: SQLiteConnector) -> None:
    """Test quoting values for DuckDB."""
    assert sqlite_connector.quote("test") == "'test'", "Value should be quoted."


def test_sqlite_connector_connect(sqlite_connector: SQLiteConnector) -> None:
    """Test connection to SQLiteDB."""
    sqlite_connector.connect()
    assert sqlite_connector.conn is not None, "Connection should be established."


def test_sqlite_connector_get_tables(sqlite_connector: SQLiteConnector) -> None:
    """Test retrieving tables and their schema."""
    sqlite_connector.connect()
    tables = sqlite_connector.get_tables()
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
    sqlite_connector: SQLiteConnector, sql: str, expected_len: int
) -> None:
    """Test various SQL queries."""
    sqlite_connector.connect()
    df = sqlite_connector.run_sql_to_df(sql)
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame."
    assert len(df) == expected_len, "DataFrame should contain two rows."
    assert list(df.columns) == ["id", "name"], "DataFrame should have correct columns."
