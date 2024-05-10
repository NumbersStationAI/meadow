"""DuckDB cache implementation."""

import duckdb

from meadow.cache.cache import Cache


class DuckDBCache(Cache):
    """A DuckDB cache."""

    def __init__(self, cache_file: str | None = None) -> None:
        """Create DuckDB cache."""
        self.cache_file = cache_file if cache_file else ".duckdb.cache"
        self.conn = duckdb.connect(self.cache_file)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)

    def close(self) -> None:
        """Close the cache."""
        self.conn.close()

    def get_key(self, key: str) -> str | None:
        """
        Get the value for a key.

        Returns None if key is not in cache.

        Args:
            key: Key for cache.
        """
        result = self.cur.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return result[0] if result else None

    def set_key(self, key: str, value: str) -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: Key for cache.
            value: New value for key.
        """
        self.cur.begin()
        self.cur.execute(
            "INSERT INTO cache (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )
        self.commit()

    def get_all_keys(self) -> list[str]:
        """
        Get all keys in cache.

        Returns:
            List of keys in cache.
        """
        result = self.cur.execute("SELECT key FROM cache").fetchall()
        return [row[0] for row in result]

    def commit(self) -> None:
        """Commit any changes to the database."""
        self.cur.commit()
