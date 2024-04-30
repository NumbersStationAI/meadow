from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, model_validator


class Column(BaseModel):
    """Column in a table."""

    """Name of the column."""
    name: str

    """Data type of the column."""
    data_type: str

    """Sample values in the column."""
    sample_values: list[Any] = None

    """Whether this column is a primary key"""
    primary_key: bool = False

    """The table and column name this column references"""
    foreign_key: tuple[str, str] = None


class Table(BaseModel):
    """Table in the database."""

    """Name of the table."""
    name: str

    """If view or not."""
    is_view: bool = False

    """Columns in the table if base table."""
    columns: list[Column] = None

    """View sql definition if view."""
    view_sql: str = None

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
        pass

    @abstractmethod
    def connect(self) -> None:
        """Connect to the database."""
        pass

    @abstractmethod
    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        pass

    @abstractmethod
    def get_tables(self) -> list[Table]:
        """Get the tables in the database."""
        pass
