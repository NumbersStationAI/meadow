"""Executor agent."""

import logging
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent
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

DEFAULT_EXECUTOR_PROMPT = """You are a error describing assistant who needs to help a user understand their error message and give hints for fixing.

Below is the data schema the user is working with.
{serialized_schema}

Given the user's message below, please explain the error and hypothesis how to fix. In free form text, summarize your thinking that explains the error and suggests possible fixes."""

ERROR_MESSAGE = """I'm sorry, I'm having a hard time running. Please try to rephrase."""


class ReaskExecutorAgent(ExecutorAgent):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[str, str, Database], AgentMessage],
        max_execution_attempts: int = 2,
        system_prompt: str = DEFAULT_EXECUTOR_PROMPT,
        reask_suffix: str = "\n\nPlease fix.",
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
        self._reask_suffix = reask_suffix
        self._messages = MessageHistory()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return f"{self._execution_func.__name__}_Executor"

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
        print(self.execution_func)
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
        except Exception as e:
            parsed_response = AgentMessage(
                role="assistant",
                content=str(e) + self._reask_suffix,
                requires_response=True,
                generating_agent=self.name,
            )
        if self._current_execution_attempts >= self._max_execution_attempts:
            # This is the final response to the supervisor so set response to False
            parsed_response.requires_response = False
            return parsed_response
        self._current_execution_attempts += 1
        # The validator w/ LLM client will summarize error and ask for help
        if self.llm_client is not None and parsed_response.requires_response:
            # Executors are a bit weird in that they are going to generate an "assistant" message
            # given the output of an execution. The last role of messages in `user`. However,
            # we need to add the error in the executor and then generate the `assistant` response.
            # Basically, this requires swapping roles in messages.
            messages_copy = []
            for m in messages:
                m_copy = m.model_copy()
                m_copy.role = "user" if m.role == "assistant" else "assistant"
                messages_copy.append(m_copy)
            parsed_response.role = "user"
            messages_copy.append(parsed_response)
            chat_response = await generate_llm_reply(
                client=self.llm_client,
                messages=messages_copy,
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
                requires_response=True,
                generating_agent=self.name,
            )
        else:
            return parsed_response
