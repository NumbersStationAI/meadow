from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel, model_validator


class Column(BaseModel):
    """Column in a table."""

    name: str
    """Name of the column."""

    data_type: str | None = None
    """Data type of the column."""

    primary_key: bool = False
    """Whether this column is a primary key"""

    foreign_keys: list[tuple[str, int]] | None = None
    """The table name and column index this column references"""


class Table(BaseModel):
    """Table in the database."""

    name: str
    """Name of the table."""

    description: str | None = None
    """Description of the table."""

    is_view: bool = False
    """If view or not."""

    is_draft: bool = False
    """If the view is a draft or not.

    Draft mode is when an executor is iterating on a view
    but it is not finalized yet. Finalized means it will appear
    in the schema.
    """

    is_deprecated: bool = False
    """If the table is deprecated or not.

    Deprecated tables are not shown in the schema or used.

    Deprecated tables are mainly base tables that were remapping or renamed or replaced by view with added columns.
    """

    is_hidden: bool = False
    """If the table is hidden or not.

    Hidden tables are those not selected for a sequence of steps in a single plan.
    As soon as the plan is done, the table will be shown again.
    """

    columns: list[Column] | None = None
    """Columns in the table if base table."""

    data: list[dict] | None = None
    """Data."""

    view_sql: str | None = None
    """View sql definition if view."""

    class Config:
        """Pydantic configuration."""

        extra = "forbid"

    @model_validator(mode="after")
    def check_valid_view_or_base_table(self) -> "Table":
        """Check that a view has view sql and a base table has columns."""
        if self.is_view and not self.view_sql:
            raise ValueError("View must have view_sql defined.")
        if not self.is_view and not self.columns:
            raise ValueError("Base table must have columns defined.")
        return self

    @model_validator(mode="after")
    def check_only_view_is_draft(self) -> "Table":
        """Check that only views can be drafts."""
        if self.is_draft and not self.is_view:
            raise ValueError("Only views can be drafts.")
        return self

    @model_validator(mode="after")
    def check_only_base_table_deprecated(self) -> "Table":
        """Check that only base tables can be deprecated."""
        if self.is_deprecated and self.is_view:
            raise ValueError("Only base tables can be deprecated.")
        return self


class Connector(ABC):
    """Abstract base class for database connectors."""

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Get the dialect of the database."""
        pass  # pragma: no cover

    @abstractmethod
    def quote(self, value: str) -> str:
        """Quote a value."""
        pass  # pragma: no cover

    @abstractmethod
    def connect(self) -> None:
        """Connect to the database."""
        pass  # pragma: no cover

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the database."""
        pass  # pragma: no cover

    @abstractmethod
    def run_sql_to_df(self, sql: str) -> pd.DataFrame:
        """Run an SQL query."""
        pass  # pragma: no cover

    @abstractmethod
    def execute_sql(self, sql: str, parameters: Any = None) -> None:
        """Run the SQL without returning anything."""
        pass  # pragma: no cover

    @abstractmethod
    def get_tables(self) -> list[Table]:
        """Get the tables in the database."""
        pass  # pragma: no cover
