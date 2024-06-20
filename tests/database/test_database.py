import pandas as pd
import pytest

from meadow.database.connector.connector import Column, Table
from meadow.database.connector.duckdb import DuckDBConnector
from meadow.database.database import (
    Database,
    _get_base_remapping_sql,
    add_base_table_remaps_as_ctes,
    add_views_as_ctes,
    check_if_non_select_query,
    extract_columns_from_df,
    get_non_matching_fks,
    map_dtype_to_sql,
    validate_sql,
)


def test_validate_sql() -> None:
    valid_sql = "SELECT * FROM users"
    invalid_sql = "SELEC * FROM users"
    assert validate_sql(valid_sql, "sqlite") == (True, None)
    assert validate_sql(invalid_sql, "sqlite") == (
        False,
        "Required keyword: 'expression' missing for <class 'sqlglot.expressions.Mul'>. Line 1, Col: 12.\n  SELEC * \x1b[4mFROM\x1b[0m users",
    )


def test_get_base_remapping_sql() -> None:
    base_table = Table(
        name="base",
        columns=[Column(name="a", data_type="int"), Column(name="b", data_type="int")],
    )
    new_table = Table(
        name="new",
        columns=[Column(name="x", data_type="int"), Column(name="y", data_type="int")],
    )
    sql = _get_base_remapping_sql(base_table, new_table, quote_func=lambda x: f"'{x}'")
    expected_sql = "SELECT base.'a' AS 'x', base.'b' AS 'y' FROM base"
    assert sql == expected_sql


def test_add_views_as_ctes() -> None:
    sql = "SELECT * FROM main_table JOIN view1 ON main_table.id = view1.id"
    views = [("view1", "SELECT id FROM table1"), ("view2", "SELECT id FROM table2")]
    result_sql = add_views_as_ctes(sql, views)
    assert (
        """WITH view1 AS (
  SELECT
    id
  FROM table1
)
SELECT
  *
FROM main_table
JOIN view1
  ON main_table.id = view1.id"""
        == result_sql
    )


def test_add_base_table_remaps_as_ctes() -> None:
    base_table = Table(
        name="base",
        columns=[Column(name="a", data_type="int"), Column(name="b", data_type="int")],
    )
    new_table = Table(
        name="base",
        columns=[Column(name="x", data_type="int"), Column(name="y", data_type="int")],
    )
    base_tables = {"base": base_table}
    base_table_remapping = {"base": new_table}

    sql = "SELECT x FROM base"
    result_sql = add_base_table_remaps_as_ctes(
        sql, base_table_remapping, base_tables, quote_func=lambda x: f'"{x}"'
    )
    assert (
        """WITH base_ AS (
  SELECT
    base."a" AS "x",
    base."b" AS "y"
  FROM base
)
SELECT
  x
FROM base_"""
        == result_sql
    )


def test_map_dtype_to_sql() -> None:
    assert map_dtype_to_sql("int64") == "INTEGER"
    assert map_dtype_to_sql("float64") == "REAL"
    assert map_dtype_to_sql("object") == "TEXT"
    assert map_dtype_to_sql("bool") == "INTEGER"
    assert map_dtype_to_sql("datetime64[ns]") == "TEXT"
    assert map_dtype_to_sql("unknown") == "TEXT"


def test_extract_columns_from_df() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    columns = extract_columns_from_df(df)
    assert columns == [
        Column(name="a", data_type="INTEGER"),
        Column(name="b", data_type="TEXT"),
    ]

    # Test duplicate columns
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "b": [3, 4]})  # noqa: F601
    columns = extract_columns_from_df(df)
    assert columns == [
        Column(name="a", data_type="INTEGER"),
        Column(name="b", data_type="INTEGER"),
    ]


def test_get_non_matching_fks() -> None:
    base_table = Table(
        name="base",
        columns=[
            Column(name="a", data_type="int", foreign_keys=[("new", 0)]),
            Column(name="b", data_type="int"),
        ],
    )
    new_table = Table(
        name="new",
        columns=[Column(name="a", data_type="int"), Column(name="y", data_type="int")],
    )
    all_tables = {"base": base_table, "new": new_table}
    assert get_non_matching_fks(all_tables) == []

    # Now test when the tables have mismatched names
    base_table = Table(
        name="base",
        columns=[
            Column(name="a", data_type="int", foreign_keys=[("new", 0)]),
            Column(name="b", data_type="int"),
        ],
    )
    new_table = Table(
        name="new",
        columns=[Column(name="x", data_type="int"), Column(name="y", data_type="int")],
    )
    all_tables = {"base": base_table, "new": new_table}
    assert get_non_matching_fks(all_tables) == [("base.a", "new.x")]


def test_check_if_non_select_query() -> None:
    assert check_if_non_select_query("CREATE TABLE test (id INT)") is True
    assert check_if_non_select_query("DROP TABLE test") is True
    assert check_if_non_select_query("UPDATE test SET id = 1") is True
    assert check_if_non_select_query("INSERT INTO test VALUES (1)") is True
    assert check_if_non_select_query("DELETE FROM test") is True
    assert check_if_non_select_query("ALTER TABLE test ADD COLUMN name TEXT") is True
    assert (
        check_if_non_select_query("ALTER TABLE test ALTER COLUMN name SET NOT NULL")
        is True
    )
    assert check_if_non_select_query("SELECT * FROM test") is False
    assert check_if_non_select_query("SELECT id FROM test WHERE name='blah';") is False


def test_database_initialization(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    assert db._connector == duckdb_connector
    assert db._base_tables == {
        "users": Table(
            name="users",
            columns=[
                Column(name="id", data_type="INTEGER"),
                Column(name="username", data_type="VARCHAR"),
                Column(name="age", data_type="INTEGER"),
            ],
            data=[
                {"id": 1, "username": "alice", "age": -10},
                {"id": 2, "username": "bob", "age": 25},
            ],
        ),
        "emails": Table(
            name="emails",
            columns=[
                Column(name="user_id", data_type="INTEGER"),
                Column(name="email", data_type="VARCHAR"),
            ],
            data=[
                {"user_id": 1, "email": "aliceeatscheese@gmail.com"},
                {"user_id": 2, "email": "bobswimswithsharks@gmail.com"},
                {"user_id": 3, "email": "martiaisahorse@gmail.com"},
            ],
        ),
    }
    assert db._view_tables == {}
    assert db._base_table_remapping == {}


def test_add_view(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT id+2 AS id2, username FROM users WHERE age < 0"
    db.add_view("user_view", sql)
    assert "user_view" in db._view_tables
    assert db._view_tables["user_view"] == Table(
        name="user_view",
        columns=[
            Column(name="id2", data_type="INTEGER"),
            Column(name="username", data_type="TEXT"),
        ],
        data=[{"id2": 3, "username": "alice"}],
        view_sql=sql,
        is_view=True,
        is_draft=True,
    )


def test_run_sql_to_df(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT * FROM users"
    df = db.run_sql_to_df(sql)
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "username": ["alice", "bob"],
            "age": [-10, 25],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_create_temp_table(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT * FROM users"
    with pytest.raises(ValueError) as e:
        db.create_temp_table(sql)
    assert str(e.value) == "Only CREATE TEMPORARY TABLE statements are allowed."

    sql = "CREATE TABLE temp AS SELECT * FROM users"
    with pytest.raises(ValueError) as e:
        db.create_temp_table(sql)
    assert str(e.value) == "Only CREATE TEMPORARY TABLE statements are allowed."

    sql = "CREATE TEMPORARY TABLE temp AS SELECT * FROM users"
    db.create_temp_table(sql)
    assert db.run_sql_to_df("SELECT * FROM temp").shape == (2, 3)


def test_insert_values_temp_table(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "CREATE TEMPORARY TABLE temp (id INTEGER, val TEXT)"
    db.create_temp_table(sql)
    assert db.run_sql_to_df("SELECT * FROM temp").empty
    db.insert_values_temp_table(
        "temp",
        [
            {"id": 1, "val": "alice"},
            {"id": 2, "val": "bob"},
            {"id": 3, "val": "charlie"},
        ],
    )
    df = db.run_sql_to_df("SELECT * FROM temp")
    expected_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "val": ["alice", "bob", "charlie"],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_finalize_draft_views(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT id+2 AS id2, username FROM users WHERE age < 0"
    db.add_view("user_view", sql)
    assert db._view_tables["user_view"].is_draft
    db.finalize_draft_views()
    assert not db._view_tables["user_view"].is_draft


def test_deprecate_table(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    assert not db._base_tables["users"].is_deprecated
    assert not db._base_tables["emails"].is_deprecated
    db.deprecate_table("users")
    assert db._base_tables["users"].is_deprecated
    assert not db._base_tables["emails"].is_deprecated


def test_hide_and_unhide_tables(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    assert not db._base_tables["users"].is_hidden
    assert not db._base_tables["emails"].is_hidden
    db.hide_table("users")
    assert db._base_tables["users"].is_hidden
    assert not db._base_tables["emails"].is_hidden
    db.unhide_all_tables()
    assert not db._base_tables["users"].is_hidden
    assert not db._base_tables["emails"].is_hidden
    db.hide_all_but("users")
    assert not db._base_tables["users"].is_hidden
    assert db._base_tables["emails"].is_hidden


def test_add_base_table_column_remap(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    assert not db._base_tables["users"].is_deprecated
    db.add_base_table_column_remap("users", {"id": "user_id"})
    assert db._base_table_remapping["users"].columns[0].name == "user_id"
    assert db._base_table_remapping["users"].columns[1].name == "username"
    assert db._base_table_remapping["users"].columns[2].name == "age"
    assert db._base_tables["users"].is_deprecated


def test_remove_base_table_remaps(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    db.add_base_table_column_remap("users", {"id": "user_id"})
    assert db._base_tables["users"].is_deprecated
    assert db._base_table_remapping
    db.remove_base_table_remaps()
    assert not db._base_tables["users"].is_deprecated
    assert not db._base_table_remapping


def test_remove_view(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT id+2 AS id2, username FROM users WHERE age < 0"
    db.add_view("user_view", sql)
    db.remove_view("user_view")
    assert "user_view" not in db._view_tables


def test_get_number_of_views(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    sql = "SELECT id+2 AS id2, username FROM users WHERE age < 0"
    db.add_view("user_view", sql)
    assert db.get_number_of_views() == 1


def test_normalize_query(duckdb_connector: DuckDBConnector) -> None:
    db = Database(duckdb_connector)
    db.add_base_table_column_remap("users", {"id": "user_id"})
    sql = "SELECT user_id+2 AS user_id2, username FROM users WHERE age < 0"
    db.add_view("user_view", sql)
    sql = "SELECT user_id2 FROM user_view"
    normalized_sql = db.normalize_query(sql)
    assert (
        normalized_sql
        == """WITH users_ AS (
  SELECT
    users."id" AS "user_id",
    users."username" AS "username",
    users."age" AS "age"
  FROM users
), user_view AS (
  SELECT
    user_id + 2 AS user_id2,
    username
  FROM users_
  WHERE
    age < 0
)
SELECT
  user_id2
FROM user_view"""
    )
