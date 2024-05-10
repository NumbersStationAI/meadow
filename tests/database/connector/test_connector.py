import pytest

from meadow.database.connector.connector import Column, Table


def test_view_with_no_sql() -> None:
    with pytest.raises(ValueError) as e:
        Table(name="test_view", is_view=True, view_sql=None)
    assert "View must have view_sql defined." in str(e.value)


def test_base_table_with_no_columns() -> None:
    with pytest.raises(ValueError) as e:
        Table(name="test_table", is_view=False, columns=[])
    assert "Base table must have columns defined." in str(e.value)


def test_valid_view() -> None:
    table = Table(name="valid_view", is_view=True, view_sql="SELECT * FROM base_table;")
    assert table.is_view
    assert table.view_sql == "SELECT * FROM base_table;"


def test_valid_table() -> None:
    column = Column(name="id", data_type="int", primary_key=True)
    table = Table(name="valid_table", is_view=False, columns=[column])
    assert not table.is_view
    assert len(table.columns) == 1
