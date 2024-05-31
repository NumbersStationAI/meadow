"""Pytest fixtures for the tests."""

import os
import tempfile
from typing import Generator

import duckdb
import pytest

from meadow.cache.cache import Cache
from meadow.cache.duckdb import DuckDBCache
from meadow.client.schema import ChatRequest, FunctionArgSpec, ToolSpec
from meadow.database.connector.duckdb import DuckDBConnector


@pytest.fixture()
def duckdb_cache() -> Generator[tuple[Cache, str], None, None]:
    """Fixture to provide a DuckDBCache instance with a temporary database."""
    file_name = tempfile.mktemp()
    test_cache = DuckDBCache(file_name)
    yield test_cache, file_name
    test_cache.close()
    os.remove(file_name)


@pytest.fixture()
def duckdb_connector() -> Generator[DuckDBConnector, None, None]:
    """Fixture to provide a DuckDBCache instance with a temporary database."""
    file_name = tempfile.mktemp()
    # Use duckdb to create a table in the database
    conn = duckdb.connect(file_name)
    conn.execute("CREATE TABLE users (id INTEGER, username VARCHAR, age INTEGER);")
    conn.execute("INSERT INTO users VALUES (1, 'alice', -10);")
    conn.execute("INSERT INTO users VALUES (2, 'bob', 25);")
    conn.execute("CREATE TABLE emails (user_id INTEGER, email VARCHAR);")
    conn.execute("INSERT INTO emails VALUES (1, 'aliceeatscheese@gmail.com');")
    conn.execute("INSERT INTO emails VALUES (2, 'bobswimswithsharks@gmail.com');")
    conn.execute("INSERT INTO emails VALUES (3, 'martiaisahorse@gmail.com');")
    conn.close()
    test_connector = DuckDBConnector(file_name)
    yield test_connector
    os.remove(file_name)


@pytest.fixture
def chat_request() -> ChatRequest:
    """Chat request."""
    return ChatRequest(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
        tools=[
            ToolSpec(
                name="query_gen",
                description="Query generator",
                function_args=[
                    FunctionArgSpec(
                        name="question",
                        description="The question to generate a query for",
                        type="string",
                        required=True,
                    )
                ],
            )
        ],
    )
