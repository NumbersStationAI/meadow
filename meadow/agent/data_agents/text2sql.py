"""SQL Generator Agent."""

import logging
from typing import Callable

from meadow.agent.agent import Agent, AgentRole, LLMAgentWithExecutors
from meadow.agent.data_agents.text2sql_utils import (
    check_empty_table,
    parse_sql_response,
)
from meadow.agent.exectors.empty_result import EmptyResultExecutor
from meadow.agent.exectors.reask import ExecutorAgent, ReaskExecutorAgent
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

DEFAULT_REASK_SUFFIX = "\n\nPlease output a single SQL query in <sql></sql> that will resolve the issue. Take your best guess as to the correct SQL."


class SQLGeneratorAgent(LLMAgentWithExecutors):
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
        self._role = AgentRole.EXECUTOR

        # Override with defaults
        if self._executors is None:
            self._executors = [
                ReaskExecutorAgent(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    execution_func=parse_sql_response,
                    reask_suffix=DEFAULT_REASK_SUFFIX,
                    llm_callback=self._llm_callback,
                ),
                EmptyResultExecutor(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    execution_func=check_empty_table,
                    reask_suffix=DEFAULT_REASK_SUFFIX,
                    llm_callback=self._llm_callback,
                ),
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

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent.

        Only used for agents that have executors."""
        self._role = role

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
        """Generate a reply when Executor agent."""
        # print(colored("SQL AGENT", "red"))
        # print(self.system_message)
        messages_start_idx = -1
        while (
            abs(messages_start_idx) <= len(messages)
            and messages[messages_start_idx].requires_response
        ):
            messages_start_idx -= 2
        print(messages_start_idx, "STARTING")
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
            requires_execution=True,
        )
