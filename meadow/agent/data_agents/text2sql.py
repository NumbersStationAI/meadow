"""SQL Generator Agent."""

import logging
import re
from typing import Callable

import sqlglot

from meadow.agent.agent import Agent, DataAgent
from meadow.agent.executor import DefaultExecutorAgent, ExecutorAgent
from meadow.agent.schema import AgentMessage, Commands
from meadow.agent.utils import (
    generate_llm_reply,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database
from meadow.database.serializer import serialize_as_xml
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_SQL_PROMPT = """You generate {dialect} SQL queries and are a {dialect} expert. Given the table schema and user's question, generate a {dialect} SQL query that answers the user's question and a one sentence description of the generated SQL query. Follow these rules:

1. Feel free to think through what you need to do first.
2. Use <description></description> tags for a one sentence description of what the table captures. Be concise.
3. Use <sql></sql> tags for the SQL. Please refer to views and base tables in the SQL if necessary.
4. If the user responds back at some point with a message that indicates the user is satisfied with the SQL (e.g. "looks good" or "continue to next step"), response ONLY with {termination_message} tags. They should NEVER be used in conjunction with <sql> or other tags.

The user's schema is:
{schema}
"""


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
            last_sql = prettify_sql(database.get_table(k).view_sql)
        except Exception as e:
            error_message = f"Failed to get last SQL from database. e={e}"
            logger.warning(error_message)
            # used to break out of try/except
            return  # type: ignore
        try:
            last_sql_df = database.run_sql_to_df(last_sql).head(5)
        except Exception as e:
            # Extract error message alone to avoid views
            err_msg = str(e).split(":", 1)[1]
            if "SELECT" in err_msg:
                print("Parsing failed")
            error_message = f"Failed to run SQL in SQLite. e={err_msg.strip()}"
            logger.warning(error_message)
            # used to break out of try/except
            return  # type: ignore
        # TODO: add error to check if DF is empty or not
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
    user_content = f"SQL:\n{last_sql}"
    if last_sql_df is not None:
        user_content += f"\n\nTable:\n{last_sql_df.to_string()}"
    return AgentMessage(
        role="assistant",
        content=content,
        display_content=user_content,
        tool_calls=None,
        generating_agent=agent_name,
    )


class SQLGeneratorAgent(DataAgent):
    """Agent that generates SQL queries from user questions."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        executors: list[ExecutorAgent] = None,
        system_prompt: str = DEFAULT_SQL_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the SQL generator agent."""
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._executors = executors
        self._system_prompt = system_prompt
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._messages = MessageHistory()

        # Override with defaults
        if self._executors is None:
            self._executors = [
                DefaultExecutorAgent(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    execution_func=parse_sql_response,
                    llm_callback=self._llm_callback,
                )
            ]

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "SQLGenerator"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Generates a single SQL query based on the given user instruction. Each instruction should be a detailed description of what attributes, aggregates, filter conditions, tables, and joins are needed in the SQL query along with any custom functions that are needed (e.g. ROW_NUMBER, RANK, LAG, ...)."

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def database(self) -> Database:
        """The database used by the agent."""
        return self._database

    @property
    def system_message(self) -> str:
        """Get the system message."""
        serialized_schema = serialize_as_xml(self.database.tables)
        return self._system_prompt.format(
            schema=serialized_schema,
            termination_message=Commands.END,
            dialect="SQLite",
        )

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executor agents that should be used by this agent."""
        return self._executors

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            logger.error("GOT EMPTY MESSAGE")
            raise ValueError("Message is empty")
        self._messages.add_message(agent=recipient, role="assistant", message=message)
        await recipient.receive(message, self)

    async def receive(
        self,
        message: AgentMessage,
        sender: Agent,
    ) -> None:
        """Receive a message from another agent."""
        if not self._silent:
            print_message(
                message,
                from_agent=sender.name,
                to_agent=self.name,
            )
        self._messages.add_message(agent=sender, role="user", message=message)

        reply = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )
        await self.send(reply, sender)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        # print(colored("SQL AGENT", "red"))
        # print(self.system_message)
        messages_start_idx = -1
        while (
            abs(messages_start_idx) <= len(messages)
            and messages[messages_start_idx].is_error_message
        ):
            messages_start_idx -= 2
        messages = messages[messages_start_idx:]
        for msg in messages:
            print(msg.role)
            print(msg.content)
        print("-------")
        print("-------")
        chat_response = await generate_llm_reply(
            client=self.llm_client,
            messages=messages,
            tools=[],
            system_message=AgentMessage(
                role="system",
                content=self.system_message,
                generating_agent=self.name,
            ),
            llm_config=self._llm_config,
            llm_callback=self._llm_callback,
            overwrite_cache=self._overwrite_cache,
        )
        content = chat_response.choices[0].message.content
        if content.endswith("<end>") and "<sql" in content:
            content = content.replace("<end>", "")
        print("SQL AGENT CONTENT", content)
        print("*****")
        return AgentMessage(
            role="assistant",
            content=content,
            tool_calls=None,
            generating_agent=self.name,
        )
