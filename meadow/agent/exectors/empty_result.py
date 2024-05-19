"""Executor agent."""

import logging
import re
from typing import Callable
from xml.etree import ElementTree

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMAgentWithExecutors
from meadow.agent.data_agents.text2sql_utils import parse_sql_response, parse_sqls
from meadow.agent.exectors.reask import ReaskExecutorAgent
from meadow.agent.schema import AgentMessage
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

DEFAULT_EMPTYRESULT_PROMPT = """You job is to help debug why a given SQL query is returning an empty table result. There's a chance nothing is wrong and it's just an empty result, but there's also a chance the query is incorrect. Common issues are incorrect WHERE clauses, incorrect use of logical operators (OR vs AND), incorrect handling of NULL values, and incorrect joins.

The user will provide their schema, question, and associated empty SQL query. You have two possible actions you can take to resolve the issue

1. [Query] Issue a query on the database of any kind.
2. [Edit] Modify the SQL.

To take an action, use the following format of
<step>
<action>Query/Edit</action>
<input>Input to action</input>
</step>

Given the user's input, provide the action you want to take. The user will provide the response and you two will collectively iterate on the issue until it is resolved. Your final goal is to edit the SQL to be correct. Please think through your decision making process first."""


def parse_plan(
    message: str,
    agent_name: str,
    database: Database,  # required for use in an executor
) -> AgentMessage:
    """Extract the plan from the response.

    Plan follows
    <step>
    <action>...</action>
    <input>...</input>
    </step>
    """
    if "<step>" not in message:
        raise ValueError(f"Plan not found in the response. message={message}")
    inner_steps = re.search(r"(<step>.*</step>)", message, re.DOTALL).group(1)
    try:
        root = ElementTree.fromstring(inner_steps)  # Parse the XML string
        for step in root:
            action = (
                step.find("action").text
                if step.find("action") is not None
                else "Unknown"
            )
            if action not in ["Query", "Edit"]:
                raise ValueError(f"Unknown action {action}. Please use Query or Edit.")
            input = (
                step.find("input").text.strip()
                if step.find("input") is not None
                else "No input"
            )
            if action == "Edit":
                input = f"<sql>/n{input}</sql>"
    except ElementTree.ParseError as e:
        error_message = f"Failed to parse the message as XML. e={e}"
        raise ValueError(error_message)
    return AgentMessage(
        role="assistant",
        content=input,
        tool_calls=None,
        generating_agent=agent_name,
        requires_response=(action == "Query"),
    )


def parse_sql_if_available(
    message: str,
    agent_name: str,
    database: Database,  # required for use in an executor
) -> AgentMessage:
    if "<sql>" in message:
        return parse_sql_response(message, agent_name, database)
    return AgentMessage(
        role="assistant",
        content=message,
        tool_calls=None,
        generating_agent=agent_name,
    )


class EmptyResultExecutor(ExecutorAgent, LLMAgentWithExecutors):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[str, str, Database], AgentMessage],
        max_execution_attempts: int = 10,
        executors: list[ExecutorAgent] = None,
        system_prompt: str = DEFAULT_EMPTYRESULT_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._execution_func = execution_func
        self._max_execution_attempts = max_execution_attempts
        self._current_execution_attempts = 0
        self._executors = executors
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._role = AgentRole.EXECUTOR

        if self._executors is None:
            self._executors = [
                ReaskExecutorAgent(
                    client=None,
                    llm_config=None,
                    database=self._database,
                    execution_func=parse_plan,
                    llm_callback=self._llm_callback,
                ),
                ReaskExecutorAgent(
                    client=None,
                    llm_config=None,
                    database=self._database,
                    execution_func=parse_sql_if_available,
                    llm_callback=self._llm_callback,
                ),
            ]

        assert (
            self._client is not None
        ), f"LLM client must be set for {self.name} to work."

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return (
            "Executes responses and asks models why there are mistakes and how to fix."
        )

    @property
    def execution_func(self) -> Callable[[str, str, Database], AgentMessage]:
        """The execution function of this agent."""
        return self._execution_func

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent.

        Only used for agents that have executors."""
        self._role = role

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executor agents that should be used by this agent."""
        return self._executors

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        return self._system_prompt

    def reset_execution_attempts(self) -> None:
        """Reset the number of execution attempts."""
        self._current_execution_attempts = 0

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
        if self.execution_func is None:
            raise ValueError(
                "Execution function is not set. Executor must have an execution function."
            )
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

    async def generate_executed_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate the reply when acting as the Executor for an Agent."""
        if self.execution_func is None:
            raise ValueError(
                "Execution function is not set. Executor must have an execution function."
            )
        try:
            parsed_response = self.execution_func(
                messages[-1].content,
                self.name,
                self._database,
            )
        except Exception:
            sql_dict = parse_sqls(messages[-1].content)
            view_table_sql = self._database.get_table(
                max(sql_dict.keys(), key=lambda x: int(x[3:]))
            ).view_sql
            assert view_table_sql, "SQL view should already have been added."
            # The error is that table is empty
            error_message = f"""My schema is:
{serialize_as_xml(self._database.tables)}

Here is the SQL query I have written:
<sql>
{view_table_sql}
</sql>

This SQL returned an empty result. Help me debug."""
            parsed_response = AgentMessage(
                role="assistant",
                content=error_message,
                requires_response=True,
                generating_agent=self.name,
            )
        if self._current_execution_attempts >= self._max_execution_attempts:
            # This is the final response to the supervisor so set response to False
            parsed_response.requires_response = False
            return parsed_response
        self._current_execution_attempts += 1
        return parsed_response

    async def generate_agent_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate the reply when acting as an Agent of a chat."""
        # Use display content instead of content for agent chats for user
        messages = [message.model_copy() for message in messages]
        for message in messages:
            if message.role == "user":
                message.content = message.display_content
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
        print("EMTPY RESULT AGENT CONTENT", content)
        print("*****")

        return AgentMessage(
            role="assistant",
            content=content,
            tool_calls=None,
            generating_agent=self.name,
        )

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        if self._role == AgentRole.EXECUTOR:
            return await self.generate_executed_reply(messages, sender)
        return await self.generate_agent_reply(messages, sender)
