"""Executor agent."""

import logging
import re
from typing import Callable

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMAgentWithExecutors
from meadow.agent.data_agents.text2sql_utils import (
    parse_and_run_single_sql,
    parse_sql_response,
    parse_sqls,
)
from meadow.agent.executor.reask import ReaskExecutor
from meadow.agent.schema import AgentMessage
from meadow.agent.utils import (
    generate_llm_reply,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database
from meadow.database.serializer import serialize_as_list, serialize_as_xml
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_DEBUGGER_PROMPT = """You job is to help debug why a given SQL query is erroring. There's a chance nothing is wrong, but there's also a chance the query is incorrect.

The user will provide their schema and associated errored SQL query. You have two possible actions you can take to resolve the issue

1. [Query] Issue a query on the database of any kind.
2. [Edit] Modify the SQL.
3. [Do Nothing] If you believe the SQL is correct and the table is appropriately empty.

To take an action, use the following format of
<step>
<action>Query/Edit/Do Nothing</action>
<input>Input to action</input>
</step>

At each point in the conversation, you must provide exaction one action you want to take. The user will provide the response and you two will collectively iterate on the issue until it is resolved. Your final goal is to edit the SQL to be correct or do nothing. If you would like, using <thinking> tags to plan the action to take before outputting the <step> tags."""


def parse_plan(
    message: str,
) -> tuple[str, str]:
    """Extract the plan from the response.

    Plan follows
    <step>
    <action>...</action>
    <input>...</input>
    </step>
    """
    # if "<step>" not in message:
    #     raise ValueError(
    #         f"message={message}. Please output a plan in a <step> tag with <action> and <input> tags. Action must be Query, Edit, or Do Nothing."
    #     )
    # inner_steps = re.search(r"(<step>.*</step>)", message, re.DOTALL).group(1)
    # pattern = re.compile(
    #     r"<step>\s*<action>(.*?)</action>\s*(<input>(.*?)</input>){0,1}\s*</step>",
    #     re.DOTALL,
    # )
    # matches = pattern.findall(inner_steps)
    # if not matches:
    #     raise ValueError(
    #         "Failed to parse the message. Outputs needs to be in <step> tags."
    #     )
    # action, _, input = matches[0]
    print(message)
    if "---" in message:
        inner_steps = message.split("---")[1]
    else:
        inner_steps = message
    if "Input:" not in inner_steps:
        action = inner_steps.split("Action:", 1)[1].strip()
        assert action == "Do Nothing", f"Action is {action}. Please use Do Nothing."
        input = ""
    else:
        action, input = inner_steps.split("Action:", 1)[1].split("Input:", 1)
    action = action.strip()
    input = input.strip().strip("```sql").strip("```")
    print("ACTION", action, "INPUT", input)
    if action not in ["Query", "Edit", "Do Nothing"]:
        raise ValueError(
            f"Unknown action {action}. Please use Query or Edit or Do Nothing."
        )
    return action, input


def parse_plan_and_take_action(
    messages: list[AgentMessage],
    agent_name: str,
    database: Database,  # required for use in an executor
    can_reask_again: bool,
) -> AgentMessage:
    """Parse generated plan and take associated action.

    E.g. parse a SQL query and run it.
    """
    content = messages[-1].content
    action, input = parse_plan(content)
    if action == "Edit":
        message = AgentMessage(
            role="assistant", content=f"<sql>\n{input}</sql>", sending_agent=agent_name
        )
        # Remove now stale view
        view_name = f"sql{database.get_number_of_views()}"
        database.remove_view(view_name)
        return parse_sql_response([message], agent_name, database, can_reask_again)
    elif action == "Query":
        message = AgentMessage(
            role="assistant", content=f"{input}", sending_agent=agent_name
        )
        response = parse_and_run_single_sql(
            [message], agent_name, database, can_reask_again
        )
        # We know that a "Query" message requires a response from this agent
        # as the query it help debug
        response.requires_response = True
        return response
    else:
        # Get the view in question as it exists in the DB
        view_name = f"sql{database.get_number_of_views()}"
        sql = database.get_table(view_name).view_sql
        return AgentMessage(
            role="assistant",
            content=f"<sql>\n{sql}</sql>",
            display_content=f"SQL:\n{sql}",
            sending_agent=agent_name,
        )


class DebuggerExecutor(ExecutorAgent, LLMAgentWithExecutors):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        execution_func: Callable[
            [list[AgentMessage], str, Database, bool], AgentMessage
        ],
        max_execution_attempts: int = 2,
        executors: list[ExecutorAgent] = None,
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
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._role = AgentRole.EXECUTOR

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
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return (
            "Executes responses and asks models why there are mistakes and how to fix."
        )

    @property
    def role(self) -> AgentRole:
        """Get the role of the agent."""
        return self._role

    @property
    def execution_func(
        self,
    ) -> Callable[[list[AgentMessage], str, Database, bool], AgentMessage]:
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
        message.receiving_agent = recipient.name
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
        can_reask_again = (
            self._current_execution_attempts < self._max_execution_attempts
        )
        parsed_response = self.execution_func(
            messages,
            self.name,
            self._database,
            can_reask_again,
        )
        if parsed_response.requires_response:
            error_message = f"""My schema is:
{serialize_as_list(self._database.tables)}

{parsed_response.content}"""
            assert len(messages) == 1
            messages[0].content = error_message
            messages[0].display_content = error_message
            return await self.generate_agent_reply(messages, sender)
        if not can_reask_again:
            # This is the final response to the supervisor so set response to False
            parsed_response.requires_response = False
            parsed_response.requires_execution = False
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
        chat_response = await generate_llm_reply(
            client=self.llm_client,
            messages=messages,
            tools=[],
            system_message=AgentMessage(
                role="system",
                content=self.system_message,
                sending_agent=self.name,
            ),
            llm_config=self._llm_config,
            llm_callback=self._llm_callback,
            overwrite_cache=self._overwrite_cache,
        )
        content = chat_response.choices[0].message.content
        print(self.system_message)
        for msg in messages:
            print(msg.role)
            print(msg.content)
            print("------")
        print("EMTPY RESULT AGENT CONTENT", content)
        print("*****")

        return AgentMessage(
            role="assistant",
            content=content,
            tool_calls=None,
            sending_agent=self.name,
            requires_execution=True,
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
