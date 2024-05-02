"""Planner agent."""

import logging
import re
import xml.etree.ElementTree as ElementTree
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import Agent, LLMAgent
from meadow.agent.schema import AgentMessage
from meadow.agent.utils import (
    generate_llm_reply,
    has_termination_condition,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
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

Below are the agents you have access to.

<agents>
{agents}
</agents>
"""


class SubTask(BaseModel):
    """Sub-task in a plan."""

    index: int
    agent: str
    prompt: str


def parse_plan(message: str) -> list[SubTask]:
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
        raise ValueError("Plan not found in the response.")
    inner_steps = re.search(r"(<steps>.*</steps>)", message, re.DOTALL).group(1)
    plan: list[SubTask] = []
    try:
        root = ElementTree.fromstring(inner_steps)  # Parse the XML string
        for step in root:
            agent = (
                step.find("agent").text if step.find("agent") is not None else "Unknown"
            )
            instruction = (
                step.find("instruction").text.strip()
                if step.find("instruction") is not None
                else "No instruction"
            )
            plan.append(SubTask(index=len(plan), agent=agent, prompt=instruction))
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
        system_prompt: str = DEFAULT_PLANNER_PROMPT,
        termination_message: str = "<exit>",
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._available_agents = {a.name: a for a in available_agents}
        self._client = client
        self._llm_config = llm_config
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        # start at -1 so when we first call move to next subtask,
        # i.e. start the task, it will be 0
        self._plan_index = -1
        self._plan: list[SubTask] = []
        self._termination_message = termination_message
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
        return self._system_prompt.format(
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
        return bool(self._plan)

    def move_to_next_agent(
        self,
    ) -> tuple[Agent | None, str | None]:
        """Move to the next agent in the task plan."""
        self._plan_index += 1
        if self._plan_index >= len(self._plan):
            logger.warning("No more sub-tasks in the plan.")
            return None, None
        sub_task = self._plan[self._plan_index]
        agent = self._available_agents[sub_task.agent]
        return agent, sub_task.prompt

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
        if "Looks good. Go on to next step." in messages[-1].content:
            return AgentMessage(
                role="assistant",
                content=self._termination_message,
                tool_calls=None,
                generating_agent=self.name,
                is_termination_message=True,
            )
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
        # print("CONTENT PLANNER", content)
        # print("*****")
        if has_termination_condition(content, self._termination_message):
            return AgentMessage(
                role="assistant",
                content=content,
                tool_calls=None,
                generating_agent=self.name,
                is_termination_message=True,
            )
        else:
            # TODO: reask to fix errors
            self._plan = parse_plan(content)
            return AgentMessage(
                role="assistant",
                content=content,
                tool_calls=None,
                generating_agent=self.name,
            )
