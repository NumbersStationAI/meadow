"""Planner agent."""

import logging
from queue import Queue
import re
import xml.etree.ElementTree as ElementTree
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import Agent, LLMAgent
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

DEFAULT_PLANNER_PROMPT = """Based on the following objective provided by the user, please break down the objective into a sequence of sub-tasks that can be solved by one the following agents. For each step sub-task in the sequence, indicate which agents should perform the task and generate a detailed instruction for the agent to follow. The user may also provide suggestions to the plan that you should take into account when generating the plan. When generating a plan, please use the following tag format to specify the plan.

<steps>
<step1>
<agent>...</agent>
<instruction>...</instruction>
</step1>
<step2>
...
</step2>
...
</steps>

If the user responds back at some point with a message that indicates the user is satisfied with the plan, please output {termination_message} and nothing else. In other words, only output a plan or {termination_message}.
{serialized_schema}
Below are the agents you have access to.

<agents>
{agents}
</agents>
"""


class SubTask:
    """Sub-task in a plan."""

    agent: Agent
    prompt: str

    def __init__(self, agent: Agent, prompt: str):
        self.agent = agent
        self.prompt = prompt


def parse_plan(message: str, available_agents: dict[str, Agent]) -> list[SubTask]:
    """Extract the plan from the response.

    Plan follows
    <steps>
    <step1>
    <agent>...</agent>
    <instruction>...</instruction>
    </step1>
    ...
    </steps>.
    """
    if "<steps>" not in message:
        raise ValueError(f"Plan not found in the response. message={message}")
    inner_steps = re.search(r"(<steps>.*</steps>)", message, re.DOTALL).group(1)
    plan: list[SubTask] = []
    try:
        root = ElementTree.fromstring(inner_steps)  # Parse the XML string
        for step in root:
            agent = available_agents[
                step.find("agent").text if step.find("agent") is not None else "Unknown"
            ]
            instruction = (
                step.find("instruction").text.strip()
                if step.find("instruction") is not None
                else "No instruction"
            )
            plan.append(SubTask(agent=agent, prompt=instruction))
    except ElementTree.ParseError:
        logger.error(f"Failed to parse the message as XML. message={message}")
    return plan


class PlannerAgent(LLMAgent):
    """Agent that generates a plan for a task."""

    def __init__(
        self,
        available_agents: list[Agent],
        client: Client,
        llm_config: LLMConfig,
        database: Database | None,
        system_prompt: str = DEFAULT_PLANNER_PROMPT,
        termination_message: str = "<exit>",
        move_on_message: str = "<next>",
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._available_agents = {a.name: a for a in available_agents}
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        # start at -1 so when we first call move to next subtask,
        # i.e. start the task, it will be 0
        self._plan: list[SubTask] = Queue()
        self._termination_message = termination_message
        self._move_on_message = move_on_message
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "Planner"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Plans the task."

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        if self._database is not None:
            serialized_schema = serialize_as_xml(self._database.tables)
            serialized_schema = f"\nBelow is the data schema the user is working with.\n{serialized_schema}\n"
        else:
            serialized_schema = ""
        return self._system_prompt.format(
            serialized_schema=serialized_schema,
            termination_message=self._termination_message,
            agents="\n".join(
                [
                    f"<agent>\n{a.name}: {a.description}\n</agent>"
                    for a in self._available_agents.values()
                ]
            ),
        )

    def has_plan(self) -> bool:
        """Check if the agent has a plan."""
        return not self._plan.empty()

    def get_next_task(
        self,
    ) -> SubTask:
        """Move to the next agent in the task plan."""
        if self._plan.empty():
            return None
        return self._plan.get()

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
        if not self._silent:
            print_message(
                message,
                from_agent=sender.name,
                to_agent=self.name,
            )
        # update the message history
        # TODO: refactor this formatting
        if len(self._messages.get_messages(sender)) == 0:
            message.content = f"<objective>{message.content}</objective>"
        else:
            message.content = f"<feedback>{message.content}</feedback>"
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

        [self._plan.put(i) for i in parse_plan(content, self._available_agents)]
        return AgentMessage(
            role="assistant",
            content=content,
            tool_calls=None,
            generating_agent=self.name,
        )
