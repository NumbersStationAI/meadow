import pytest

from meadow.database.connector.connector import Column, Table
from meadow.database.serializer import serialize_as_list, serialize_as_xml


@pytest.fixture
def tables() -> list[Table]:
    """Create a list of tables."""
    return [
        Table(
            name="users",
            columns=[
                Column(name="id", data_type="int", primary_key=True),
                Column(name="username", data_type="varchar"),
                Column(
                    name="email",
                    data_type="varchar",
                    foreign_keys=[("emails", 0)],
                ),
            ],
            data=[
                {"id": 1, "username": "alice", "email": "aliceeatscheese@gmail.com"},
                {"id": 2, "username": "bob", "email": "bobswimswithsharks@gmail.com"},
            ],
        ),
        Table(
            name="emails",
            columns=[
                Column(name="email", data_type="varchar", primary_key=True),
                Column(name="user_id", data_type="int", foreign_keys=[("users", 0)]),
            ],
        ),
    ]


@pytest.fixture
def tables_with_views() -> list[Table]:
    """Create a list of tables."""
    return [
        Table(
            name="users",
            columns=[
                Column(name="id", data_type="int", primary_key=True),
                Column(name="username", data_type="varchar"),
                Column(
                    name="email",
                    data_type="varchar",
                    foreign_keys=[("emails", 0)],
                ),
            ],
            data=[
                {"id": 1, "username": "alice", "email": "aliceeatscheese@gmail.com"},
                {"id": 2, "username": "bob", "email": "bobswimswithsharks@gmail.com"},
            ],
        ),
        Table(
            name="emails",
            columns=[
                Column(name="email", data_type="varchar", primary_key=True),
                Column(name="user_id", data_type="int", foreign_keys=[("users", 0)]),
            ],
        ),
        Table(
            name="active_users",
            is_view=True,
            view_sql="SELECT * FROM users WHERE active = 1",
        ),
        Table(
            name="inactive_users",
            columns=[
                Column(name="id", data_type="int", primary_key=True),
            ],
            is_view=True,
            is_draft=True,
            view_sql="SELECT id FROM users WHERE active = 0",
        ),
    ]


def test_serialize_as_xml_no_view(tables: list[Table]) -> None:
    xml_output = serialize_as_xml(tables)
    expected_xml = """
<schema>
  <table>
    <name>users</name>
    <column>
      <name>id</name>
      <dataType>int</dataType>
      <primaryKey>True</primaryKey>
    </column>
    <column>
      <name>username</name>
      <dataType>varchar</dataType>
      <primaryKey>False</primaryKey>
    </column>
    <column>
      <name>email</name>
      <dataType>varchar</dataType>
      <primaryKey>False</primaryKey>
      <foreignKey references="emails(email)></foreignKey>
    </column>
    <view>False</view>
  </table>
  <table>
    <name>emails</name>
    <column>
      <name>email</name>
      <dataType>varchar</dataType>
      <primaryKey>True</primaryKey>
    </column>
    <column>
      <name>user_id</name>
      <dataType>int</dataType>
      <primaryKey>False</primaryKey>
      <foreignKey references="users(id)></foreignKey>
    </column>
    <view>False</view>
  </table>
</schema>
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_xml_with_view(tables_with_views: list[Table]) -> None:
    xml_output = serialize_as_xml(tables_with_views)
    expected_xml = """
<schema>
  <table>
    <name>users</name>
    <column>
      <name>id</name>
      <dataType>int</dataType>
      <primaryKey>True</primaryKey>
    </column>
    <column>
      <name>username</name>
      <dataType>varchar</dataType>
      <primaryKey>False</primaryKey>
    </column>
    <column>
      <name>email</name>
      <dataType>varchar</dataType>
      <primaryKey>False</primaryKey>
      <foreignKey references="emails(email)></foreignKey>
    </column>
    <view>False</view>
  </table>
  <table>
    <name>emails</name>
    <column>
      <name>email</name>
      <dataType>varchar</dataType>
      <primaryKey>True</primaryKey>
    </column>
    <column>
      <name>user_id</name>
      <dataType>int</dataType>
      <primaryKey>False</primaryKey>
      <foreignKey references="users(id)></foreignKey>
    </column>
    <view>False</view>
  </table>
  <table>
    <name>active_users</name>
    <view>True</view>
    <sql>
    SELECT * FROM users WHERE active = 1
    </sql>
  </table>
  <table>
    <name>inactive_users</name>
    <column>
      <name>id</name>
      <dataType>int</dataType>
      <primaryKey>True</primaryKey>
    </column>
    <view>True</view>
    <sql>
    SELECT id FROM users WHERE active = 0
    </sql>
  </table>
</schema>
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_list_no_view(tables: list[Table]) -> None:
    xml_output = serialize_as_list(tables)
    expected_xml = """
# users(id, username, email)
#    Data:
#       id username                         email
#    0   1    alice     aliceeatscheese@gmail.com
#    1   2      bob  bobswimswithsharks@gmail.com
#    Joins:
#    users.email = emails.email
# emails(email, user_id)
#    Joins:
#    emails.user_id = users.id
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_list_with_view(tables_with_views: list[Table]) -> None:
    xml_output = serialize_as_list(tables_with_views)
    expected_xml = """
# users(id, username, email)
#    Data:
#       id username                         email
#    0   1    alice     aliceeatscheese@gmail.com
#    1   2      bob  bobswimswithsharks@gmail.com
#    Joins:
#    users.email = emails.email
# emails(email, user_id)
#    Joins:
#    emails.user_id = users.id
# active_users()
#    ```sql
#    SELECT * FROM users WHERE active = 1
#    ```
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_list_with_draft_view(tables_with_views: list[Table]) -> None:
    xml_output = serialize_as_list(tables_with_views, skip_draft=False)
    expected_xml = """
# users(id, username, email)
#    Data:
#       id username                         email
#    0   1    alice     aliceeatscheese@gmail.com
#    1   2      bob  bobswimswithsharks@gmail.com
#    Joins:
#    users.email = emails.email
# emails(email, user_id)
#    Joins:
#    emails.user_id = users.id
# active_users()
#    ```sql
#    SELECT * FROM users WHERE active = 1
#    ```
# inactive_users(id)
#    ```sql
#    SELECT id FROM users WHERE active = 0
#    ```
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_list_with_view_skip_constraints(
    tables_with_views: list[Table],
) -> None:
    xml_output = serialize_as_list(tables_with_views, add_constraints=False)
    expected_xml = """
# users(id, username, email)
#    Data:
#       id username                         email
#    0   1    alice     aliceeatscheese@gmail.com
#    1   2      bob  bobswimswithsharks@gmail.com
# emails(email, user_id)
# active_users()
#    ```sql
#    SELECT * FROM users WHERE active = 1
#    ```
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"


def test_serialize_as_list_with_view_skip_data(tables_with_views: list[Table]) -> None:
    xml_output = serialize_as_list(tables_with_views, add_data=False)
    expected_xml = """
# users(id, username, email)
#    Joins:
#    users.email = emails.email
# emails(email, user_id)
#    Joins:
#    emails.user_id = users.id
# active_users()
#    ```sql
#    SELECT * FROM users WHERE active = 1
#    ```
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"
