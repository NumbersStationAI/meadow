"""Pytest fixtures for the tests."""

import os
import tempfile
from typing import Generator

import pytest

from meadow.cache.cache import Cache
from meadow.cache.duckdb import DuckDBCache
from meadow.client.schema import ChatRequest, ToolSpec


@pytest.fixture()
def duckdb_cache() -> Generator[tuple[Cache, str], None, None]:
    """Fixture to provide a DuckDBCache instance with a temporary database."""
    file_name = tempfile.mktemp()
    test_cache = DuckDBCache(file_name)
    yield test_cache, file_name
    test_cache.close()
    os.remove(file_name)


@pytest.fixture
def chat_request():
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
                    {
                        "name": "question",
                        "description": "The question to generate a query for",
                        "type": "string",
                        "required": True,
                    }
                ],
            )
        ],
    )
