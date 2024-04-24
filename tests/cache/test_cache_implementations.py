import os
import tempfile
from typing import Any, Generator

import pytest

from meadow.cache.cache import Cache
from meadow.cache.duckdb import DuckDBCache
from meadow.cache.sqllite import SQLiteCache


@pytest.fixture
def sqlite_cache() -> Generator[tuple[Cache, str], None, None]:
    """Fixture to provide a SQLiteCache instance with a temporary database."""
    file_name = tempfile.mktemp()
    test_cache = SQLiteCache(file_name)
    yield test_cache, file_name
    test_cache.close()
    os.remove(file_name)


@pytest.fixture(params=["duckdb_cache", "sqlite_cache"])
def cache_fixture(request: Any) -> tuple[Cache, str]:
    """Fixture to provide a Cache instance."""
    if request.param == "duckdb_cache":
        return request.getfixturevalue("duckdb_cache")
    elif request.param == "sqlite_cache":
        return request.getfixturevalue("sqlite_cache")
    else:
        raise ValueError("Invalid cache type.")


def test_set_and_get_key(cache_fixture: tuple[Cache, str]) -> None:
    """Test setting and getting a key-value pair."""
    cache, _ = cache_fixture
    cache.set_key("test_key", "test_value")
    assert (
        cache.get_key("test_key") == "test_value"
    ), "The value should match what was set."


def test_get_nonexistent_key(cache_fixture: tuple[Cache, str]) -> None:
    """Test retrieval of a nonexistent key."""
    cache, _ = cache_fixture
    assert (
        cache.get_key("nonexistent") is None
    ), "Should return None for nonexistent keys."


def test_overwrite_existing_key(cache_fixture: tuple[Cache, str]) -> None:
    """Test overwriting an existing key's value."""
    cache, _ = cache_fixture
    cache.set_key("test_key", "initial_value")
    cache.set_key("test_key", "new_value")
    assert (
        cache.get_key("test_key") == "new_value"
    ), "Should overwrite the existing value."


def test_get_all_keys(cache_fixture: tuple[Cache, str]) -> None:
    """Test retrieval of all keys."""
    cache, _ = cache_fixture
    cache.set_key("key1", "value1")
    cache.set_key("key2", "value2")
    cache.set_key("key3", "value3")
    assert cache.get_all_keys() == [
        "key1",
        "key2",
        "key3",
    ], "Should return all keys in the cache."


@pytest.mark.parametrize(
    "cache_fixture", ["duckdb_cache", "sqlite_cache"], indirect=True
)
def test_close_connection(cache_fixture: tuple[Cache, str]) -> None:
    """Test closing the database connection."""
    cache, _ = cache_fixture
    cache.set_key("test_key", "test_value")
    cache.close()
    with pytest.raises(Exception):
        cache.get_key(
            "test_key"
        )  # Attempting to use the cache after closing should raise an exception


@pytest.mark.parametrize("cache_type", ["duckdb_cache", "sqlite_cache"])
def test_commit_changes(
    cache_type: str, duckdb_cache: tuple[Cache, str], sqlite_cache: tuple[Cache, str]
) -> None:
    """Test commit functionality."""
    if cache_type == "duckdb_cache":
        cache, cache_file = duckdb_cache
        class_generator: type[DuckDBCache] | type[SQLiteCache] = DuckDBCache
    elif cache_type == "sqlite_cache":
        cache, cache_file = sqlite_cache
        class_generator = SQLiteCache
    cache.set_key("test_key", "test_value")
    cache.commit()
    # Reconnect to the database to ensure the commit was successful
    # get class of cache
    new_cache = class_generator(cache_file)
    assert (
        new_cache.get_key("test_key") == "test_value"
    ), "Should persist changes across sessions."
    new_cache.close()
