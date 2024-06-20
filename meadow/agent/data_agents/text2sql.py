"""SQL Generator Agent."""

import logging
from typing import Callable

from meadow.agent.agent import (
    Agent,
    AgentRole,
    ExecutorAgent,
    LLMAgentWithExecutors,
)
from meadow.agent.executor.data_executors.empty_result_debugger import (
    EmptyResultExecutor,
)
from meadow.agent.executor.data_executors.sql_validate_reask import (
    SQLValidateExecutor,
)
from meadow.agent.schema import AgentMessage, ClientMessageRole
from meadow.agent.utils import (
    generate_llm_reply,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database
from meadow.database.serializer import serialize_as_list
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_SQL_PROMPT = """You generate SQLite SQL queries and are a SQLite expert. Given the table schema and user's question, generate a SQLite SQL query that answers the user's question and a one sentence description of the generated SQL query. Follow these rules:

1. First use <sql></sql> tags or ```sql...``` for the SQL. Please refer to views and base tables in the SQL if necessary.
2. Then <description></description> tags for a one sentence description of what the SQL query captures. Be concise.

The user's schema is:
{schema}"""

DEFAULT_SQL_DESC = "Generates a single SQL query based on the given user instruction. Each instruction should clearly describe what question is to be asked and what attributes the user wants."


class SQLGeneratorAgent(LLMAgentWithExecutors):
    """Agent that generates SQL queries from user questions."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        executors: list[ExecutorAgent] = None,
        name: str = "SQLGenerator",
        description: str = DEFAULT_SQL_DESC,
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
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._messages = MessageHistory()
        self._role = AgentRole.TASK_HANDLER

        if self._executors is None:
            self._executors = [
                SQLValidateExecutor(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    llm_callback=self._llm_callback,
                ),
                EmptyResultExecutor(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    llm_callback=self._llm_callback,
                ),
            ]

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

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
        serialized_schema = serialize_as_list(self.database.tables)
        return self._system_prompt.format(
            schema=serialized_schema,
        )

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent.

        Only used for agents that have executors."""
        self._role = role

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executor agents that should be used by this agent."""
        return self._executors

    def get_messages(self, chat_agent: "Agent") -> list[AgentMessage]:
        """Get the messages between self and the chat_agent."""
        return self._messages.get_messages(chat_agent)

    def add_to_messages(
        self, chat_agent: "Agent", messages: list[AgentMessage]
    ) -> None:
        """Add chat messages between self and chat_agent.

        Used when starting hierarchical chats and historical messages
        need to be passed to the agent.
        """
        self._messages.copy_messages_from(chat_agent, messages)

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            raise ValueError("Message is empty")
        self._messages.add_message(
            agent=recipient, agent_role=ClientMessageRole.SENDER, message=message
        )
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
        self._messages.add_message(
            agent=sender, agent_role=ClientMessageRole.RECEIVER, message=message
        )

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
        # If the SQL generator is called multiple times, we want to start
        # from the last new SQL command and ignore previous SQL queries.
        # But, we want to keep any feedback messages from validators or the user
        # So, we keep messages in pairs that require respnose (from validator)
        messages_start_idx = -1
        while (
            abs(messages_start_idx) <= len(messages)
            and messages[messages_start_idx].requires_response
        ):
            messages_start_idx -= 2
        messages = messages[messages_start_idx:]
        chat_response = await generate_llm_reply(
            client=self.llm_client,
            messages=messages,
            tools=[],
            system_message=AgentMessage(
                agent_role=ClientMessageRole.SYSTEM,
                content=self.system_message,
                sending_agent=self.name,
            ),
            llm_config=self._llm_config,
            llm_callback=self._llm_callback,
            overwrite_cache=self._overwrite_cache,
        )
        content = chat_response.choices[0].message.content
        if "```sql" in content:
            if "<sql>" in content:
                content = content.replace("```sql", "").replace("```", "")
            else:
                content = content.replace("```sql", "<sql>").replace("```", "</sql>")
        if content.endswith("<end>") and "<sql" in content:
            content = content.replace("<end>", "")
        return AgentMessage(
            content=content,
            sending_agent=self.name,
            requires_execution=True,
        )
