"""SQLLite cache implementation."""

import sqlite3

from meadow.cache.cache import Cache


class SQLiteCache(Cache):
    """A sqllite cache."""

    def __init__(self, cache_file: str | None = None) -> None:
        """Create SQLite cache."""
        self.cache_file = cache_file if cache_file else ".sqlite.cache"
        self.sql = sqlite3.connect(self.cache_file)
        self.cursor = self.sql.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.commit()

    def close(self) -> None:
        """Close the cache."""
        self.sql.close()

    def get_key(self, key: str) -> str | None:
        """
        Get the value for a key.

        Returns None if key is not in cache.

        Args:
            key: Key for cache.
        """
        self.cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set_key(self, key: str, value: str) -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: Key for cache.
            value: New value for key.
        """
        self.cursor.execute(
            "REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
        )

    def get_all_keys(self) -> list[str]:
        """
        Get all keys in cache.

        Returns:
            List of keys in cache.
        """
        self.cursor.execute("SELECT key FROM cache")
        return [row[0] for row in self.cursor.fetchall()]

    def commit(self) -> None:
        """Commit any changes to the database."""
        self.sql.commit()
