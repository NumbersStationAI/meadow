"""Serialize the database utils."""

from meadow.database.connector.connector import Table


def serialize_as_xml(tables: list[Table]) -> str:
    """Serialize the database using XML tags."""
    xml_parts = ["<schema>"]
    for table in tables:
        xml_parts.append(f'  <table name="{table.name}" view="{table.is_view}">')
        if table.columns:
            for column in table.columns:
                xml_line = f'    <column name="{column.name}" type="{column.data_type}"'
                if column.primary_key:
                    xml_line += ' primaryKey="true"'
                if column.foreign_keys:
                    for fk_table, fk_column in column.foreign_keys:
                        xml_line += (
                            f' foreignKey="true" references="{fk_table}({fk_column})"'
                        )
                xml_line += "/>"
                xml_parts.append(xml_line)
        else:
            xml_parts.append("    <sql>\n    " + table.view_sql + "\n    </sql>")
        xml_parts.append("  </table>")
    xml_parts.append("</schema>")
    return "\n".join(xml_parts)
