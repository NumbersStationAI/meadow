"""Executor agent."""

import functools
import json
import logging
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent, LLMAgentWithExecutors
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

DEFAULT_BATCH_LLM_PROMPT = """You are a data analyst assistant who is going to update examples of data one at a time. Given a user instruction and a row of data, please output the correct value for the new column."""

DEFAULT_BATCH_LLM_DESC = "Runs a model to perform a task on each row of data."


class BatchLLMExecutor(ExecutorAgent, LLMAgentWithExecutors):
    """Agent that execute/validates a response on each row of a table.

    For this agent to work, it requires 1 table in the database. It also
    requires that the parsed_response from the execution_func be a string
    that can be formatted with the input_val parameter.
    """

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[ExecutorFunctionInput], AgentMessage],
        max_execution_attempts: int = 2,
        executors: list[ExecutorAgent] = None,
        description: str = DEFAULT_BATCH_LLM_DESC,
        system_prompt: str = DEFAULT_BATCH_LLM_PROMPT,
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
        self._description = description
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._role = AgentRole.TASK_HANDLER

        if self._executors is None:
            self._executors = []

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        if isinstance(self._execution_func, functools.partial):
            return f"{self._execution_func.func.__name__}_BatchLLM_Executor"
        else:
            return f"{self._execution_func.__name__}_BatchLLM_Executor"

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
    def database(self) -> Database:
        """The database used by the agent."""
        return self._database

    @property
    def system_message(self) -> str:
        """Get the system message."""
        assert (
            len(self.database.tables) == 1
        ), "Only one table should be in the database for column generation."
        serialized_schema = serialize_as_list(self.database.tables[:1])
        return self._system_prompt.format(serialized_schema=serialized_schema)

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
        """Generate a reply based on the received messages."""
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
            return parsed_response
        self._current_execution_attempts += 1
        if parsed_response.requires_response:
            # Executors are a bit weird in that they are going to generate an "assistant" message
            # given the output of an execution. The last role of messages in `user`. However,
            # we need to add the error in the executor and then generate the `assistant` response.
            # Basically, this requires swapping roles in messages.
            messages_copy = []
            for m in messages:
                m_copy = m.model_copy()
                m_copy.agent_role = (
                    ClientMessageRole.RECEIVER
                    if m.agent_role == ClientMessageRole.SENDER
                    else ClientMessageRole.SENDER
                )
                # Set the message role, too (this is typically done internally)
                m_copy.role = m_copy.agent_role.value
                messages_copy.append(m_copy)
            parsed_response.agent_role = ClientMessageRole.RECEIVER
            parsed_response.role = parsed_response.agent_role.value
            # For the batch llm call agent, we need to iterate over rows of the given
            # table and generate a response for each row. To pass the response into
            # the prompt, we use `input_val` as a parameter in the prompt.
            assert (
                len(self.database.tables) == 1
            ), "Only one table should be in the database for batch llm execution."
            table_to_iter = self.database.tables[0]
            all_rows = self.database.run_sql_to_df(
                f"SELECT * FROM {table_to_iter.name} ORDER BY {', '.join(map(lambda x: x.name, table_to_iter.columns))}"
            )
            all_new_col_values = []
            for _, row in all_rows.iterrows():
                parsed_response_copy = parsed_response.model_copy()
                parsed_response_copy.content = parsed_response_copy.content.format(
                    input_val=row.to_dict()
                )
                messages_copy.append(parsed_response_copy)
                chat_response = await generate_llm_reply(
                    client=self.llm_client,
                    messages=messages_copy,
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
                print("Batch Executor", content)
                all_new_col_values.append(content)
                messages_copy.pop()
            final_content = json.dumps(all_new_col_values)
            return AgentMessage(
                content=final_content,
                requires_response=False,
                requires_execution=True,
                sending_agent=self.name,
            )
        else:
            return parsed_response
