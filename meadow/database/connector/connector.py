from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, model_validator


class Column(BaseModel):
    """Column in a table."""

    name: str
    """Name of the column."""

    data_type: str
    """Data type of the column."""

    sample_values: list[Any] | None = None
    """Sample values in the column."""

    primary_key: bool = False
    """Whether this column is a primary key"""

    foreign_keys: list[tuple[str, str]] | None = None
    """The table and column name this column references"""


class Table(BaseModel):
    """Table in the database."""

    name: str
    """Name of the table."""

    is_view: bool = False
    """If view or not."""

    columns: list[Column] | None = None
    """Columns in the table if base table."""

    view_sql: str | None = None
    """View sql definition if view."""

    @model_validator(mode="after")
    def check_valid_view_or_base_table(self) -> "Table":
        """Check that a view has view sql and a base table has columns."""
        if self.is_view and not self.view_sql:
            raise ValueError("View must have view_sql defined.")
        if not self.is_view and not self.columns:
            raise ValueError("Base table must have columns defined.")
        return self


class Connector(ABC):
    """Abstract base class for database connectors."""

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Get the dialect of the database."""
        pass  # pragma: no cover

    @abstractmethod
    def connect(self) -> None:
        """Connect to the database."""
        pass  # pragma: no cover

    @abstractmethod
    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        pass  # pragma: no cover

    @abstractmethod
    def get_tables(self) -> list[Table]:
        """Get the tables in the database."""
        pass  # pragma: no cover
