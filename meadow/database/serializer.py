"""Serialize the database utils."""

import pandas as pd

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
                    for fk_table, fk_column_idx in column.foreign_keys:
                        fk_table_obj = [t for t in tables if t.name == fk_table][0]
                        xml_line += f'\n      <foreignKey references="{fk_table}({fk_table_obj.columns[fk_column_idx].name})></foreignKey>'
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


def serialize_as_list(
    tables: list[Table],
    skip_draft: bool = True,
    add_constraints: bool = True,
    add_data: bool = True,
) -> str:
    """Serialize in the format # table(attr,...), one per line."""
    # TODO: Add view finalization
    serialized_tables = []
    for table in tables:
        if skip_draft and table.is_draft:
            continue
        serialized_table = f"# {table.name}("
        serialized_table += ", ".join([column.name for column in table.columns or []])
        serialized_table += ")"
        if table.description:
            serialized_table += f": {table.description}"
        if table.view_sql:
            view_sql_lines = table.view_sql.split("\n")
            view_sql = "\n#    ".join(view_sql_lines)
            serialized_table += f"\n#    ```sql\n#    {view_sql}\n#    ```"
        if add_data and table.data:
            df = pd.DataFrame(table.data)
            serialized_table += "\n#    Data:"
            table_lines = df.to_string().split("\n")
            serialized_table += "\n#    " + "\n#    ".join(table_lines)
        if add_constraints:
            join_str = ""
            for column in table.columns or []:
                # if column.primary_key:
                #     serialized_table += f"\n#    PRIMARY KEY({column.name})"
                if column.foreign_keys:
                    for fk_table, fk_column_idx in column.foreign_keys:
                        fk_table_obj = [t for t in tables if t.name == fk_table][0]
                        join_str += f"\n#    {table.name}.{column.name} = {fk_table}.{fk_table_obj.columns[fk_column_idx].name}"
            if join_str:
                serialized_table += "\n#    Joins:" + join_str
        serialized_tables.append(serialized_table)
    return "\n".join(serialized_tables)
