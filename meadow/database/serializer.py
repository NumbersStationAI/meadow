"""Serialize the database utils."""

from meadow.database.connector.connector import Table


def serialize_as_xml(tables: list[Table]) -> str:
    """Serialize the database using XML tags."""
    xml_parts = ["<schema>"]
    for table in tables:
        xml_parts.append("  <table>")
        xml_parts.append(f"    <name>{table.name}</name>")
        if table.description:
            xml_parts.append(f"    <description>{table.description}</description>")
        if table.columns:
            for column in table.columns:
                xml_line = "    <column>"
                xml_line += f"\n      <name>{column.name}</name>"
                if column.data_type:
                    xml_line += f"\n      <dataType>{column.data_type}</dataType>"
                xml_line += f"\n      <primaryKey>{column.primary_key}</primaryKey>"
                if column.foreign_keys:
                    for fk_table, fk_column in column.foreign_keys:
                        xml_line += f'\n      <foreignKey references="{fk_table}({fk_column})></foreignKey>'
                xml_line += "\n    </column>"
                xml_parts.append(xml_line)
        xml_parts.append(f"    <view>{table.is_view}</view>")
        if table.view_sql:
            # Add 4 spaces to each line of the view sql
            view_sql_lines = table.view_sql.split("\n")
            view_sql = "\n    ".join(view_sql_lines)
            xml_parts.append(f"    <sql>\n    {view_sql}\n    </sql>")
        xml_parts.append("  </table>")
    xml_parts.append("</schema>")
    return "\n".join(xml_parts)
