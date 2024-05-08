"""Executor agent."""

import logging
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent, LLMAgent
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

DEFAULT_EXECUTOR_PROMPT = """You are a error solving assistant who needs to help a user figure out their error message and take the appropriate action to fix.

Below is the data schema the user is working with.
{serialized_schema}

Given the user's message below, please explain why the query is wrong and then give a few options for how to fix the query. Make sure to think step-by-step about what the error means and the best solutions in <thinking></thinking> tags.
"""

ERROR_MESSAGE = """I'm sorry, I'm having a hard time running. Please try to rephrase."""


class DefaultExecutorAgent(ExecutorAgent, LLMAgent):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[str, str, str, Database], AgentMessage],
        max_execution_attempts: int = 2,
        system_prompt: str = DEFAULT_EXECUTOR_PROMPT,
        termination_message: str = "<exit>",
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
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._termination_message = termination_message
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "Executor"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return (
            "Executes responses and asks models why there are mistakes and how to fix."
        )

    @property
    def execution_func(self) -> Callable[[str, str, str, Database], AgentMessage]:
        """The execution function of this agent."""
        return self._execution_func

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        serialized_schema = serialize_as_xml(self._database.tables)
        return self._system_prompt.format(serialized_schema=serialized_schema)

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

        reply, to_send = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )
        await self.send(reply, to_send)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply based on the received messages."""
        if self.execution_func is None:
            raise ValueError(
                "Execution function is not set. Executor must have an execution function."
            )
        if self._current_execution_attempts >= self._max_execution_attempts:
            # Do not set as error message because this is the final response to the user
            return AgentMessage(
                role="assistant",
                content=ERROR_MESSAGE,
                tool_calls=None,
                generating_agent=self.name,
            ), sender
        self._current_execution_attempts += 1
        try:
            parsed_response = self.execution_func(
                messages[-1].content,
                self.name,
                self._termination_message,
                self._database,
            )
        except Exception as e:
            parsed_response = AgentMessage(
                role="assistant",
                content=str(e) + "\n\nPlease fix.",
                is_error_message=True,
                generating_agent=self.name,
            )
        # The validator w/ LLM client will summarize error and ask for help
        if self.llm_client is not None and parsed_response.is_error_message:
            # Add error to message
            messages[-1].content += f"\n\nError:\n{parsed_response.content}"
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
            return AgentMessage(
                role="assistant",
                content=content,
                tool_calls=None,
                generating_agent=self.name,
            ), sender
        else:
            return parsed_response, sender
