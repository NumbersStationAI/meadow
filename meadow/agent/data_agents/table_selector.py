"""Agent that outputs the user desired table."""

import logging
import re
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent, LLMAgentWithExecutors
from meadow.agent.executor.user_feedback import UserFeedbackExecutor
from meadow.agent.schema import (
    AgentMessage,
    AgentRole,
    ClientMessageRole,
    ExecutorFunctionInput,
)
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

DEFAULT_ATTRIBUTE_PROMPT = """The user wants to answer an question or perform a task over some data.

Below is the data schema the user is working with.
{schema}

Given the user's question, output the single table that the user wants to analyze. If you are unsure what table is best, please output all the options and get feedback from the user as to which one is best.

For example, suppose the user is asking over a table with columns people(uid, name, last_name, age, country) and employment(eid, uid, cid, job, salary) and companies(cid, name, country).

Question: Generate a user's full name
Explanation: The user wants to know the full name of the person from the people table.
Table: people

Question: Show the user id at their company
Explanation: The user wants to know the user id of the person at their company from the employment table.
Table: employment

Question: What is the name
Explanation: The user wants to know the name of the company from the companies table or the user name from the people table.
Table: companies
Table: people

Now output an explanation followed by the table the user wants to see for the following question:"""


DEFAULT_ATTRIBUTE_DESC = "Subselects the best table to use for a user's task. This is required to be used before generating new columns. The instruction to this agent should be a copy of the user question or input."


def parse_for_user_feedback(
    input: ExecutorFunctionInput,
) -> AgentMessage:
    content = input.messages[-1].content
    # Get the tables
    table_statements = re.findall(r"Table: (:?.+)\n", content)
    if not table_statements:
        final_content = "I am having a hard time knowing what table you want to use. Can you please provide more information?"
        requires_response = True
    elif len(table_statements) > 1:
        final_content = (
            f"Which of the following tables would you like to use?\n{table_statements}"
        )
        requires_response = True
    elif table_statements[0] not in [tbl.name for tbl in input.database.tables]:
        final_content = f"The table {table_statements[0]} is not in the database."
        requires_response = True
    else:
        final_content = table_statements[0]
        requires_response = False
        # Now subselect the table in the database
        input.database.hide_all_but(table_statements[0])
    return AgentMessage(
        content=final_content,
        sending_agent=input.agent_name,
        requires_response=requires_response,
    )


class SingleTableSelectorAgent(LLMAgentWithExecutors):
    """Agent that selects a single SQL table."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        executors: list[ExecutorAgent] = None,
        name: str = "SingleTableSelectorAgent",
        description: str = DEFAULT_ATTRIBUTE_DESC,
        system_prompt: str = DEFAULT_ATTRIBUTE_PROMPT,
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
                UserFeedbackExecutor(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    execution_func=parse_for_user_feedback,
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
        return AgentMessage(
            content=content,
            sending_agent=self.name,
            requires_execution=True,
        )
