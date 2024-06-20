"""Executor agent."""

import copy
import logging
from typing import Callable

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMAgentWithExecutors
from meadow.agent.data_agents.text2sql_utils import (
    parse_and_run_sql_for_debugger,
    parse_sql_response,
)
from meadow.agent.executor.reask import ReaskExecutor
from meadow.agent.schema import AgentMessage, ClientMessageRole, ExecutorFunctionInput
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

DEFAULT_DEBUGGER_PROMPT = """You job is to help debug why a given SQL query is erroring. There's a chance nothing is wrong, but there's also a chance the query is incorrect.

The user will provide their schema and associated empty SQL query. You have three possible actions you can take to resolve the issue

1. [Query] Issue a single SELECT SQL query on the database of any kind.
2. [Edit] Modify the SQL.
3. [Do Nothing] If you believe the SQL is correct and the table is appropriately empty.

You must always output one action of the following form:

---
Action: Query/Edit/Do Nothing
Input: ```Input to action in quotes```
---

At each point in the conversation, you must provide exaction one action you want to take. The user will provide the response and you two will collectively iterate on the issue until it is resolved. Your final goal is to edit the SQL to be correct or do nothing. If you would like, using <thinking> tags to plan the action to take before outputting the <step> tags."""


DEFAULT_DEBUGGER_DESC = (
    "Executes responses and asks models why there are mistakes and how to fix."
)


def parse_plan(
    message: str,
) -> tuple[str, str]:
    """Extract the plan from the response."""
    if "---" in message:
        inner_steps = message.split("---")[1]
    else:
        inner_steps = message
    if "Action:" not in inner_steps:
        raise ValueError(
            "Action is not in the message. Please use\nAction: Query, Edit, or Do Nothing\nInput: ```Input to action in quotes```."
        )
    if "Input:" not in inner_steps:
        action = inner_steps.split("Action:", 1)[1].strip()
        assert action == "Do Nothing", f"Action is {action}. Please use Do Nothing."
        input = ""
    else:
        action, input = inner_steps.split("Action:", 1)[1].split("Input:", 1)
    action = action.strip()
    input = input.strip().strip("```sql").strip("```")
    if action not in ["Query", "Edit", "Do Nothing"]:
        raise ValueError(
            f"Unknown action {action}. Please use Query or Edit or Do Nothing."
        )
    return action, input


def parse_plan_and_take_action(
    input: ExecutorFunctionInput,
) -> AgentMessage:
    """Parse generated plan and take associated action.

    E.g. parse a SQL query and run it.
    """
    content = input.messages[-1].content
    try:
        action, input_inst = parse_plan(content)
    except ValueError as e:
        assert input.can_reask_again, "TODO: Handle this case."
        return AgentMessage(
            content=f"Error parsing the plan.\n{e}",
            requires_response=True,
            sending_agent=input.agent_name,
        )
    if action == "Edit":
        message = AgentMessage(
            content=f"<sql>\n{input_inst}</sql>",
            sending_agent=input.agent_name,
        )
        # Remove now stale view
        view_name = f"sql{input.database.get_number_of_views()}"
        input.database.remove_view(view_name)

        input_copy = copy.copy(input)
        input_copy.messages = [message]
        return parse_sql_response(input_copy)
    elif action == "Query":
        message = AgentMessage(
            content=f"{input_inst}",
            sending_agent=input.agent_name,
        )

        input_copy = copy.copy(input)
        input_copy.messages = [message]
        response = parse_and_run_sql_for_debugger(input_copy)
        # We know that a "Query" message requires a response from this agent
        # as the query it help debug
        response.requires_response = True
        return response
    else:
        # Get the view in question as it exists in the DB
        view_name = f"sql{input.database.get_number_of_views()}"
        sql = input.database.get_table(view_name).view_sql
        return AgentMessage(
            content=f"<sql>\n{sql}</sql>",
            display_content=f"SQL:\n{sql}",
            sending_agent=input.agent_name,
        )


class DebuggerExecutor(ExecutorAgent, LLMAgentWithExecutors):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[[ExecutorFunctionInput], AgentMessage],
        max_execution_attempts: int = 2,
        executors: list[ExecutorAgent] = None,
        description: str = DEFAULT_DEBUGGER_DESC,
        system_prompt: str = DEFAULT_DEBUGGER_PROMPT,
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
            # N.B. when an executor errors, it is that executor that gets resent a response
            # from the supervisor. We don't go "up" the executor chain.
            self._executors = [
                ReaskExecutor(
                    client=None,
                    llm_config=None,
                    database=self._database,
                    execution_func=parse_plan_and_take_action,
                    max_execution_attempts=15,
                    llm_callback=self._llm_callback,
                ),
            ]

        assert (
            self._client is not None
        ), f"LLM client must be set for {self.name} to work."

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return f"{self._execution_func.__name__}_Debugger_Executor"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

    @property
    def role(self) -> AgentRole:
        """Get the role of the agent."""
        return self._role

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

    async def generate_task_handler_reply(
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
            # Adding the data to the schema for the debugger makes it often think it's seeing the entire table
            # and make mistakes wrt filters. Better to leave the data out.
            error_message = f"""My schema is:
{serialize_as_list(self._database.tables, add_data=False)}

{parsed_response.content}"""
            assert len(messages) == 1
            messages[0].content = error_message
            messages[0].display_content = error_message
            return await self.generate_agent_reply(messages, sender)
        self._current_execution_attempts += 1
        return parsed_response

    async def generate_agent_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate the reply when acting as an Agent of a chat."""
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

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        if self._role == AgentRole.TASK_HANDLER:
            return await self.generate_task_handler_reply(messages, sender)
        return await self.generate_agent_reply(messages, sender)
