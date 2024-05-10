import pytest

from meadow.database.connector.connector import Column, Table
from meadow.database.serializer import serialize_as_xml


@pytest.fixture
def tables() -> list[Table]:
    """Create a list of tables."""
    return [
        Table(
            name="users",
            columns=[
                Column(name="id", data_type="int", primary_key=True),
                Column(
                    name="username", data_type="varchar", sample_values=["alice", "bob"]
                ),
                Column(
                    name="email",
                    data_type="varchar",
                    foreign_keys=[("emails", "email")],
                ),
            ],
        ),
        Table(
            name="emails",
            columns=[
                Column(name="email", data_type="varchar", primary_key=True),
                Column(name="user_id", data_type="int", foreign_keys=[("users", "id")]),
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
                Column(
                    name="username", data_type="varchar", sample_values=["alice", "bob"]
                ),
                Column(
                    name="email",
                    data_type="varchar",
                    foreign_keys=[("emails", "email")],
                ),
            ],
        ),
        Table(
            name="emails",
            columns=[
                Column(name="email", data_type="varchar", primary_key=True),
                Column(name="user_id", data_type="int", foreign_keys=[("users", "id")]),
            ],
        ),
        Table(
            name="active_users",
            is_view=True,
            view_sql="SELECT * FROM users WHERE active = 1",
        ),
    ]


def test_serialize_as_xml_no_view(tables: list[Table]) -> None:
    xml_output = serialize_as_xml(tables)
    expected_xml = """
<schema>
  <table name="users" view="False">
    <column name="id" type="int" primaryKey="true"/>
    <column name="username" type="varchar"/>
    <column name="email" type="varchar" foreignKey="true" references="emails(email)"/>
  </table>
  <table name="emails" view="False">
    <column name="email" type="varchar" primaryKey="true"/>
    <column name="user_id" type="int" foreignKey="true" references="users(id)"/>
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
  <table name="users" view="False">
    <column name="id" type="int" primaryKey="true"/>
    <column name="username" type="varchar"/>
    <column name="email" type="varchar" foreignKey="true" references="emails(email)"/>
  </table>
  <table name="emails" view="False">
    <column name="email" type="varchar" primaryKey="true"/>
    <column name="user_id" type="int" foreignKey="true" references="users(id)"/>
  </table>
  <table name="active_users" view="True">
    <sql>
    SELECT * FROM users WHERE active = 1
    </sql>
  </table>
</schema>
"""
    assert (
        xml_output.strip() == expected_xml.strip()
    ), "XML serialization should match expected output"
