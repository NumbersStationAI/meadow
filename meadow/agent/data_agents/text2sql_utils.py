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


def parse_sqls(message: str) -> list[str]:
    """Extract the SQL query from the response."""
    # find <sql1>, <sql2>, ... tags and return the SQL query
    # this will return a list of tuples where the tuples are the groups
    sql_components = re.findall(r"(<sql\d*>(.*?)<\/sql\d*>)", message, re.DOTALL)
    if not sql_components:
        raise ValueError("SQL not found in the response.")
    sqls = []
    for sql_pair in sql_components:
        sql_with_tag, sql = sql_pair
        sql = sql.replace(";", "").strip()
        sqls.append(sql)
    return sqls


def parse_description(message: str) -> str | None:
    """Extract the description from the response."""
    # find <description></description> tags and return the description
    description_components = re.findall(
        r"(<description>(.*?)<\/description>)", message, re.DOTALL
    )
    if not description_components:
        return None
    return description_components[0][1].strip()


# TODO mark as executor function
def parse_sql_response(
    messages: list[AgentMessage],
    agent_name: str,
    database: Database,
    can_reask_again: bool,
) -> AgentMessage:
    """Generate a parsed response from the SQL query."""
    reask_suffix = "\n\nPlease output a SQL query in that will resolve the issue. Take your best guess as to the correct SQL."
    content = messages[-1].content
    error_message: str = None
    final_view_sql: str = None
    added_views = set()
    try:
        next_sql_id = f"sql{database.get_number_of_views() + 1}"
        sqls = parse_sqls(content)
        if len(sqls) != 1:
            logger.warning(
                f"Multiple SQL queries found in response. sql={sqls}. Taking last one."
            )
        sql_to_add = sqls[-1]
        description = parse_description(content)
    except Exception as e:
        error_message = f"Failed to parse SQL in response. e={e}"
        logger.warning(error_message)

    if not error_message:
        try:
            view_table = database.get_table(f"sql{database.get_number_of_views()}")
            # Sometimes planner will give the same step twice for simple queries.
            if view_table and view_table.view_sql == sql_to_add:
                print("NO IDEA")
            database.add_view(name=next_sql_id, sql=sql_to_add, description=description)
            added_views.add(next_sql_id)
        except Exception as e:
            error_message = f"Failed to add view to database. e={e}"
            logger.warning(error_message)

    if not error_message:
        try:
            final_view_sql = prettify_sql(database.get_table(next_sql_id).view_sql)
        except Exception as e:
            error_message = f"Failed to get last SQL from database. e={e}"
            logger.warning(error_message)

    if not error_message:
        try:
            database.run_sql_to_df(final_view_sql).head(5)
        except Exception as e:
            if ":" in str(e):
                err_msg = str(e).split(":", 1)[1]
                if "SELECT" in err_msg:
                    print("Parsing failed")
            else:
                err_msg = str(e)
            error_message = f"Failed to run SQL in SQLite. e={err_msg.strip()}"
            logger.warning(error_message)

    # If error message, we must either respond for a LLM to fix (if can_reask_again)
    # or we must gracefully fail so the next steps can move.
    if error_message:
        for k in added_views:
            database.remove_view(k)
        if can_reask_again:
            return AgentMessage(
                role="assistant",
                content=error_message + reask_suffix,
                requires_response=True,
                sending_agent=agent_name,
            )
        else:
            if final_view_sql:
                failure_content = f"SQL:\n{final_view_sql}\n\nWarning:\n{error_message}"
            else:
                print("???")
                failure_content = None
                # Find first content in messages that isn't requiring response
                for m in messages[::-1]:
                    if not m.requires_response:
                        failure_content = m.content
                        break
                if not failure_content:
                    print("GIVING UP")
                    failure_content = content
            return AgentMessage(
                role="assistant",
                # Content is what gets fed to future models. All code relies on the
                # parsing from this content.
                content=content,
                # Display content is what goes to user
                display_content=failure_content,
                sending_agent=agent_name,
                # This will be final response to supervisor so no response required
                requires_response=False,
            )

    return AgentMessage(
        role="assistant",
        content=content,
        display_content=f"SQL:\n{final_view_sql}",
        sending_agent=agent_name,
        requires_response=False,
    )


# TODO mark as executor function
def parse_and_run_single_sql(
    messages: list[AgentMessage],
    agent_name: str,
    database: Database,
    can_reask_again: bool,
) -> AgentMessage:
    """Generate a parsed response from the SQL query."""
    content = messages[-1].content
    sql = prettify_sql(content.strip())
    error_message = None
    try:
        df = database.run_sql_to_df(sql).head(5)
    except Exception as e:
        # Extract error message alone to avoid views
        if ":" in str(e):
            err_msg = str(e).split(":", 1)[1]
        else:
            err_msg = str(e)
        if "SELECT" in err_msg:
            print("Parsing failed")
        error_message = f"Failed to run SQL in SQLite. e={err_msg.strip()}"
    if error_message:
        if can_reask_again:
            return AgentMessage(
                role="assistant",
                content=error_message,
                sending_agent=agent_name,
                requires_response=True,
            )
        else:
            return AgentMessage(
                role="assistant",
                # Content is what gets fed to future models. All code relies on the
                # parsing from this content.
                content=content,
                # Display content is what goes to user
                display_content=f"SQL:\n{sql}\n\nWarning:\n{error_message}",
                sending_agent=agent_name,
                requires_response=False,
            )
    return AgentMessage(
        role="assistant",
        content=f"SQL:\n{sql}\n\nTable:\n{df.to_string()}",
        sending_agent=agent_name,
    )


# TODO mark as executor function
def check_empty_table(
    messages: list[AgentMessage],
    agent_name: str,
    database: Database,
    can_reask_again: bool,
) -> AgentMessage:
    """Check if the resulting parsed SQL has an empty table.

    THIS WILL BE CALLED ON THE OUTPUT OF parse_sql_response.
    """
    reask_suffix = "\n\nIf you think there is a bug in the query, please output a single SQL query in that will resolve the issue. Take your best guess as to the correct SQL. Otherwise, output only the same query again. Only output one query."
    content = messages[-1].content
    error_message = None
    pretty_sql = None
    final_description = None
    sql_df = None
    try:
        existing_sql_id = f"sql{database.get_number_of_views()}"
        sqls = parse_sqls(content)
        description = parse_description(content)
        if len(sqls) != 1:
            logger.warning(
                f"Multiple SQL queries found in response. sql={sqls}. Taking last one."
            )
        pretty_sql_from_sqls = prettify_sql(sqls[-1])
        # The view should have been added from parse_sql_response
        # If it wasn't, that means the SQL failed to run
        view_table = database.get_table(existing_sql_id)
        if not view_table or not view_table.view_sql:
            # Just take the sql from the content
            pretty_sql = pretty_sql_from_sqls
            final_description = description
        else:
            pretty_sql = prettify_sql(view_table.view_sql)
            final_description = view_table.description
        # This means this is a reask and the SQL has changed
        if pretty_sql != pretty_sql_from_sqls:
            try:
                database.run_sql_to_df(pretty_sql)
                database.remove_view(existing_sql_id)
                database.add_view(
                    name=existing_sql_id,
                    sql=pretty_sql_from_sqls,
                    description=view_table.description,
                )
                print("Changed\n", pretty_sql, "\n\nTO....", pretty_sql_from_sqls)
                print("NO IDEA")
                pretty_sql = pretty_sql_from_sqls
            except Exception as e:
                logger.warning(e)
                pass
        sql_df = database.run_sql_to_df(pretty_sql).head(10)
    except Exception as e:
        error_message = (
            f"Question: {final_description}\nFailed to parse SQL in response. e={e}"
        )
        logger.warning(error_message)

    if not error_message:
        # Check if a count(*) query with a 0
        count_zero = sql_df.shape == (1, 1) and sql_df.values[0][0] == 0
        if sql_df.empty or all(sql_df.isnull().all()) or count_zero:
            assert pretty_sql
            # Keep display content in error message for debugging and because
            # the empty table may not be a real error
            error_message = f"Question: {final_description}\n```sql\n{pretty_sql}\n```\nThis SQL query generated an empty table."

    # If error message, we must either respond for a LLM to fix (if can_reask_again)
    # or we must gracefully fail so the next steps can move.
    if error_message:
        if can_reask_again:
            return AgentMessage(
                role="assistant",
                content=error_message + reask_suffix,
                requires_response=True,
                sending_agent=agent_name,
            )
        else:
            if pretty_sql:
                if sql_df is not None:
                    failure_content = f"Question: {final_description}\nSQL:\n{pretty_sql}\n\nTable:\n{sql_df.to_string()}\n\nWarning:\n{error_message}"
                else:
                    failure_content = f"Question: {final_description}\nSQL:\n{pretty_sql}\n\nWarning:\n{error_message}"
            else:
                print("NO IDEA")
                failure_content = None
                # Find first content in messages that isn't requiring response
                for m in messages[::-1]:
                    if not m.requires_response:
                        failure_content = m.content
                        break
                if not failure_content:
                    print("GIVING UP")
                    failure_content = content
            return AgentMessage(
                role="assistant",
                # Content is what gets fed to future models. All code relies on the
                # parsing from this content.
                content=content,
                # Display content is what goes to user
                display_content=failure_content,
                sending_agent=agent_name,
                # This will be final response to supervisor so no response required
                requires_response=False,
            )
    display_content = f"SQL:\n{pretty_sql}"
    display_content += f"\n\nTable:\n{sql_df.to_string()}"

    return AgentMessage(
        role="assistant",
        content=content,
        display_content=display_content,
        sending_agent=agent_name,
        requires_response=False,
    )
