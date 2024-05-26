"""SQL decomposer agent."""

import json
import logging
import re
from queue import Queue
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import (
    Agent,
    LLMPlannerAgent,
    SubTask,
)
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.schema import AgentMessage, Commands
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

DEFAULT_SQL_DECOMP_PROMPT = """The user wants to answer an analytics question in SQL.

Based on the following question provided by the user, please make a plan consisting of a minimal sequence of SQL-queries using SQL Generator agent. The SQL Generator generates a single SQL query based on the given user instructions.

For each SQL in the sequence, generate a text instruction for the SQL Generator to follow. To reference a past step (e.g. to reference step 2, please use the phrase `sql2` in the instruction). When generating a plan, please use the following tag format to specify the plan.

<instruction1>...</instruction1>
<instruction2>...</instruction2>
...

Below is the data schema the user is working with.
{serialized_schema}

Please keep the plan simple and use as few steps as possible. The last instruction should specifically say what the final attributes are. The last instruction tag should end with a sentence starting with "The final attributes should be"... and then provide the attributes."""


class SubTaskForParse(BaseModel):
    """Sub-task in a plan used in executor."""

    agent_name: str
    prompt: str


def parse_steps(input_str: str) -> list[tuple[str, str]]:
    """
    Parse the given XML-like string and extract instruction using regular expressions.
    """
    # Use '.*' after last instruction because sometimes it'll add extra pieces we don't care about
    pattern = re.compile(
        r"<instruction\d*>(.*?)</instruction\d*>",
        re.DOTALL,
    )
    matches = pattern.findall(input_str)
    return [instruction.strip() for instruction in matches]


def parse_plan(
    message: str,
    agent_name: str,
    agent_to_use: str = "SQLGenerator",
) -> AgentMessage:
    """Extract the plan from the response."""
    parsed_steps = parse_steps(message)
    plan: list[SubTaskForParse] = []
    for instruction in parsed_steps:
        plan.append(SubTaskForParse(agent_name=agent_to_use, prompt=instruction))
    return AgentMessage(
        role="assistant",
        content=json.dumps([m.model_dump() for m in plan]),
        display_content=message,
        tool_calls=None,
        sending_agent=agent_name,
        requires_response=False,
    )


class SQLDecomposerAgent(LLMPlannerAgent):
    """Agent that generates a plan for subsql tasks."""

    def __init__(
        self,
        client: Client | None,
        llm_config: LLMConfig | None,
        database: Database | None,
        available_agents: list[Agent] = None,
        system_prompt: str = DEFAULT_SQL_DECOMP_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._plan: Queue[SubTask] = Queue()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

        if available_agents is None:
            available_agents = [
                SQLGeneratorAgent(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                )
            ]
        self._available_agents = {a.name: a for a in available_agents}

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "MultiCTESQLGenerator"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "For heavily complex SQL queries that require multiple CTE expression, this agent decomposes the task into simpler sub queries that get joined together. This agent should be used instead of the simple SQLGenerator if the question is heavily complex."

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        serialized_schema = serialize_as_list(self._database.tables)
        return self._system_prompt.format(
            serialized_schema=serialized_schema,
        )

    @property
    def available_agents(self) -> dict[str, Agent]:
        """Get the available agents."""
        return self._available_agents

    def move_to_next_agent(
        self,
    ) -> SubTask:
        """Move to the next agent in the task plan."""
        if self._plan.empty():
            return None
        subtask = self._plan.get()
        # When moving on, reset executors to allow for new attempts
        if subtask.agent.executors:
            for ex in subtask.agent.executors:
                ex.reset_execution_attempts()
        return subtask

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
        if self.llm_client is not None:
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
            # print(self.system_message)
            # print(messages[-1].content)
            print("SQL DECOMP PLANNER", content)
            print("*****")
            if Commands.has_end(content):
                return AgentMessage(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                    sending_agent=self.name,
                    is_termination_message=True,
                )
            else:
                display_content = None
                try:
                    # TODO: refactor using executors
                    parsed_plan_message = parse_plan(content, self.name)
                    display_content = parsed_plan_message.display_content
                    parsed_plan = [
                        SubTaskForParse(**m)
                        for m in json.loads(parsed_plan_message.content)
                    ]
                    # If the plan is just a single step, replace with the direct question from the user with the attributes
                    if len(parsed_plan) == 1:
                        parsed_plan[0].prompt = messages[-1].content
                    for sub_task in parsed_plan:
                        self._plan.put(
                            SubTask(
                                agent=self._available_agents[sub_task.agent_name],
                                prompt=sub_task.prompt,
                            )
                        )
                except Exception as e:
                    logger.warning(
                        f"Error in parsing plan. Ignoring as executor should throw error back to fix. e={e}, message={content}."
                    )
                    raise e
                return AgentMessage(
                    role="assistant",
                    content=content,
                    display_content=display_content,
                    sending_agent=self.name,
                )
        else:
            if len(self._available_agents) > 1:
                raise ValueError("No LLM client provided and more than one agent.")
            agent = list(self._available_agents.values())[0]
            raw_content = messages[-1].content
            self._plan.put(SubTask(agent=agent, prompt=raw_content))
            serialized_plan = f"<steps><step1><agent>{agent.name}</agent><instruction>{raw_content}</instruction></step1></steps>"
            return AgentMessage(
                role="assistant",
                content=serialized_plan,
                sending_agent=self.name,
            )
