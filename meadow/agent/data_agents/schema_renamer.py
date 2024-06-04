"""Agent that cleans and renames DB schemas."""

import json
import logging
import re
from typing import Callable

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMAgentWithExecutors
from meadow.agent.executor.reask import ReaskExecutor
from meadow.agent.schema import AgentMessage, ClientMessageRole, ExecutorFunctionInput
from meadow.agent.utils import (
    generate_llm_reply,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database, get_non_matching_fks
from meadow.database.serializer import serialize_as_list
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_RENAME_PROMPT = """Your goal is to clean up a schema to make detecting joins and understanding the data easier for asking queries. You can rename the columns as you see fit.

The user will give you a schema and you need to determine what the join columns are and how, if at all, they should be renamed to be more intuitive. Join columns should be the same name and columns that do not join should be named differently.

Given the schema, first explain what columns join already. Then provide a renaming of the columns that makes the schema more intuitive where join keys match; i.e. tableA.key = tableB.key. If a column does not join, it should not be named to match another join key.

Output the remapping in JSON in the following format:
```json
{
  "table_name": {
    "old_column1_name": "old_or_new_column1_name",
    ...
  },
  "table_name": {
    "old_column1_name": "old_or_new_column1_name",
    ...
  },
}
```"""

DEFAULT_RENAME_DESC = "This agent renames columns to be more standard and useful for determining the right attributes and detecting joins. Most schemas need some cleaning unless the attributes are already very clear. The input instruction should be the phrase 'clean schema'. Do not pass the output of this agent to any other agent. Generate fresh instructions for future agents in the plan."


def parse_rename_and_update_db(
    input: ExecutorFunctionInput,
) -> AgentMessage:
    """Parse the message and update the database."""
    content_str = input.messages[-1].content
    error_message: str = None
    try:
        if "```json" in content_str:
            json_content = re.findall(r"```json\n(.*?)\n```", content_str, re.DOTALL)[
                -1
            ]
        else:
            json_content = content_str
        content: dict[str, dict[str, str]] = json.loads(json_content)
    except json.JSONDecodeError as e:
        error_message = f"The content is not a valid JSON object.\n{e}"

    for tbl, col_map in content.items():
        # Find any duplicate col map values
        if len(set(col_map.values())) != len(col_map.values()):
            # Find the exact duplicate value
            duplicate_values = set()
            for value in col_map.values():
                if value in duplicate_values:
                    error_message = (
                        f"Duplicate column name '{value}' found in table '{tbl}'."
                    )
                    break
                duplicate_values.add(value)

    if not error_message:
        for table_name, column_mapping in content.items():
            if any(k != v for k, v in column_mapping.items()):
                try:
                    input.database.add_base_table_column_remap(
                        table_name, column_mapping
                    )
                except Exception as e:
                    error_message = f"Error adding the column remapping for table {table_name}.\n{e}"
                    break

    # Make sure FKs match
    if not error_message:
        table_as_dict = {tbl.name: tbl for tbl in input.database.tables}
        non_match_pairs = get_non_matching_fks(table_as_dict)
        if non_match_pairs:
            error_message = (
                f"The following FKs do not match:\n{non_match_pairs}\nThis may be ineviatable and"
                " if so, do not worry about changing anything and output the same remapping. Otherwise, please rename."
            )

    if error_message:
        input.database.remove_base_table_remaps()
        return AgentMessage(
            content=error_message + " Please regenerate mapping and try again.",
            requires_response=True,
            sending_agent=input.agent_name,
        )

    return AgentMessage(
        content="The schema has been updated",
        display_content=f"The schema has been updated:\n{json.dumps(content, indent=2)}",
        sending_agent=input.agent_name,
    )


class SchemaRenamerAgent(LLMAgentWithExecutors):
    """Agent that generates SQL queries from user questions."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        executors: list[ExecutorAgent] = None,
        name: str = "SchemaRenamer",
        description: str = DEFAULT_RENAME_DESC,
        system_prompt: str = DEFAULT_RENAME_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the SQL generator agent."""
        self._client = client
        self._llm_config = llm_config
        self._llm_config = self._llm_config.model_copy()
        self._llm_config.max_tokens = 2000
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
                ReaskExecutor(
                    client=None,
                    llm_config=None,
                    database=self._database,
                    execution_func=parse_rename_and_update_db,
                    llm_callback=self._llm_callback,
                )
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
        return self._system_prompt

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
        messages[0].content = "My schema is\n" + serialize_as_list(self.database.tables)
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
