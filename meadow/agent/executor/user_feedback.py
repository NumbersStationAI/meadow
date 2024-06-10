"""Executor agent for user feedback."""

import logging
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent
from meadow.agent.schema import AgentMessage, ClientMessageRole, ExecutorFunctionInput
from meadow.agent.utils import (
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_USERFEEDBACK_DESC = (
    "Provides targeted feedback to model responses when model needs it."
)


class UserFeedbackExecutor(ExecutorAgent):
    """Executor that calls out to a user if asked."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[ExecutorFunctionInput], AgentMessage],
        max_execution_attempts: int = 2,
        description: str = DEFAULT_USERFEEDBACK_DESC,
        system_prompt: str = None,
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
        self._description = description
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return f"{self._execution_func.__name__}_UserFeedback_Executor"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

    @property
    def execution_func(
        self,
    ) -> Callable[[ExecutorFunctionInput], AgentMessage]:
        """The execution function of this agent."""
        return self._execution_func

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
        """Generate the reply when acting as the Executor for an Agent."""
        if self.execution_func is None:
            raise ValueError(
                "Execution function is not set. Executor must have an execution function."
            )
        can_reask_again = (
            self._current_execution_attempts < self._max_execution_attempts
        )
        execution_func_input = ExecutorFunctionInput(
            messages=messages,
            agent_name=self.name,
            database=self._database,
            can_reask_again=can_reask_again,
        )
        parsed_response = self.execution_func(execution_func_input)
        if not can_reask_again:
            # This is the final response to the supervisor so set response to False
            parsed_response.requires_response = False
            parsed_response.requires_execution = False
            return parsed_response
        if parsed_response.requires_response:
            print_message(
                parsed_response,
                from_agent=sender.name,
                to_agent=self.name,
            )
            content = input(">>> ")
            return AgentMessage(
                content=content,
                sending_agent=self.name,
                requires_response=True,
            )
        self._current_execution_attempts += 1
        return parsed_response
