"""Pytest fixtures for the tests."""

import os
import tempfile
from typing import Generator

import pytest

from meadow.cache.cache import Cache
from meadow.cache.duckdb import DuckDBCache


@pytest.fixture()
def duckdb_cache() -> Generator[tuple[Cache, str], None, None]:
    """Fixture to provide a DuckDBCache instance with a temporary database."""
    file_name = tempfile.mktemp()
    test_cache = DuckDBCache(file_name)
    yield test_cache, file_name
    test_cache.close()
    os.remove(file_name)
