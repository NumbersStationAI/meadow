import logging
import re

import sqlglot

from meadow.agent.schema import AgentMessage
from meadow.database.database import Database

logger = logging.getLogger(__name__)


def prettify_sql(sql: str, dialect: str = "sqlite") -> str:
    """Prettify the SQL query."""
    try:
        sql = sqlglot.parse_one(sql, dialect=dialect).sql(dialect=dialect, pretty=True)
    except Exception as e:
        logger.warning(f"Failed to parse SQL in {dialect} format. sql={sql}, e={e}")
        pass
    return sql


def parse_sqls(message: str) -> dict[str, str]:
    """Extract the SQL query from the response."""
    # find <sql1>, <sql2>, ... tags and return the SQL query
    # this will return a list of tuples where the tuples are the groups
    sql_components = re.findall(r"(<sql\d+>(.*?)<\/sql\d+>)", message, re.DOTALL)
    if not sql_components:
        raise ValueError(
            "SQL not found in the response. Please use <sql> tags to mark the SQL."
        )
    sql_dict = {}
    for sql_pair in sql_components:
        sql_with_tag, sql = sql_pair
        match = re.search(r"<(sql\d+)>", sql_with_tag)
        if match:
            sql = sql.replace(";", "").strip()
            sql_dict[match.group(1)] = sql
    return sql_dict


def parse_description(message: str) -> str | None:
    """Extract the description from the response."""
    # find <description></description> tags and return the description
    description_components = re.findall(
        r"(<description>(.*?)<\/description>)", message, re.DOTALL
    )
    if not description_components:
        return None
    return description_components[0][1].strip()


def handle_unnumbered_sqls(message: str, next_sql_i: int) -> str:
    """For each <sql></sql> tag that doesn't have a number, add one."""
    sql_components = re.findall(r"(<sql>(.*?)<\/sql>)", message, re.DOTALL)
    for i, sql_pair in enumerate(sql_components):
        sql_with_tag, sql = sql_pair
        message = message.replace(
            sql_with_tag, f"<sql{next_sql_i+i}>{sql}</sql{next_sql_i+i}>"
        )
    return message


def parse_sql_response(
    content: str, agent_name: str, database: Database
) -> AgentMessage:
    """Generate a parsed response from the SQL query."""
    print("CONTENT PARSE", content)
    try:
        content = handle_unnumbered_sqls(content, database.get_number_of_views() + 1)
        sql_dict = parse_sqls(content)
        assert len(sql_dict) == 1, "Only one SQL query should be generated."
        description = parse_description(content)
    except Exception as e:
        error_message = f"Failed to parse SQL in response. e={e}"
        logger.warning(error_message)
        raise ValueError(error_message)
    try:
        # update history with new SQL
        added_views = set()
        error_message = None
        k = max(sql_dict.keys(), key=lambda x: int(x[3:]))
        v = sql_dict[k]
        view_table = database.get_table(k)
        # If there is a new definition, then it's likely a reask and we should
        # update
        if view_table is None or v != view_table.view_sql:
            try:
                database.add_view(name=k, sql=v, description=description)
                added_views.add(k)
            except Exception as e:
                error_message = f"Failed to add view to database. e={e}"
                logger.warning(error_message)
                # used to break out of try/except
                return  # type: ignore
        try:
            # get the last sql and return it fully parsed
            final_view_sql = prettify_sql(database.get_table(k).view_sql)
        except Exception as e:
            error_message = f"Failed to get last SQL from database. e={e}"
            logger.warning(error_message)
            # used to break out of try/except
            return  # type: ignore
        try:
            database.run_sql_to_df(final_view_sql).head(5)
        except Exception as e:
            # Extract error message alone to avoid views
            err_msg = str(e).split(":", 1)[1]
            if "SELECT" in err_msg:
                print("Parsing failed")
            error_message = f"Failed to run SQL in SQLite. e={err_msg.strip()}"
            logger.warning(error_message)
            # used to break out of try/except
            return  # type: ignore
    except Exception as e:
        error_message = f"Failed to get last SQL from database. e={e}"
        logger.warning(error_message)
        # used to break out of try/except
        return  # type: ignore
    finally:
        if error_message:
            # Clean up possibly buggy views
            for k in added_views:
                database.remove_view(k)
            raise ValueError(error_message)
    return AgentMessage(
        role="assistant",
        content=content,
        display_content=f"SQL:\n{final_view_sql}",
        tool_calls=None,
        generating_agent=agent_name,
        requires_response=False,
    )


def parse_and_run_single_sql(
    sql: str, agent_name: str, database: Database
) -> AgentMessage:
    """Generate a parsed response from the SQL query."""
    print("CONTENT PARSE SIMPLE", sql)
    sql = prettify_sql(sql.strip())
    try:
        df = database.run_sql_to_df(sql).head(5)
    except Exception as e:
        # Extract error message alone to avoid views
        err_msg = str(e).split(":", 1)[1]
        if "SELECT" in err_msg:
            print("Parsing failed")
        error_message = f"Failed to run SQL in SQLite. e={err_msg.strip()}"
        logger.warning(error_message)
    return AgentMessage(
        role="assistant",
        content=f"SQL:{sql}\bData:\n{df.to_string()}",
        tool_calls=None,
        generating_agent=agent_name,
    )


def check_empty_table(
    content: str, agent_name: str, database: Database
) -> AgentMessage:
    """Check if the resulting parsed SQL has an empty table.

    THIS WILL BE CALLED ON THE OUTPUT OF parse_sql_response.
    """
    print("CONTENT EMPTy", content)
    try:
        sql_dict = parse_sqls(content)
        assert len(sql_dict) == 1, "Only one SQL query should be generated."
        k = max(sql_dict.keys(), key=lambda x: int(x[3:]))
        # The view will have been added from parse_sql_response
        view_table = database.get_table(k)
        assert view_table.view_sql, "SQL view should already have been added."
        pretty_sql = prettify_sql(view_table.view_sql)
        sql_df = database.run_sql_to_df(view_table.view_sql).head(10)
    except Exception as e:
        error_message = f"Failed to parse SQL in response. e={e}"
        logger.warning(error_message)
        raise ValueError(error_message)

    display_content = f"SQL:\n{pretty_sql}"
    display_content += f"\n\nTable:\n{sql_df.to_string()}"

    if sql_df.empty or all(sql_df.isnull().all()):
        # Keep display content in error message for debugging and because
        # the empty table may not be a real error
        raise ValueError(f"{display_content}\nSQL query generated an empty table.")

    return AgentMessage(
        role="assistant",
        content=content,
        display_content=display_content,
        tool_calls=None,
        generating_agent=agent_name,
        requires_response=False,
    )
