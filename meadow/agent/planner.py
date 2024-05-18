"""Planner agent."""

import json
import logging
import re
import xml.etree.ElementTree as ElementTree
from functools import partial
from queue import Queue
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import Agent, ExecutorAgent, LLMAgent
from meadow.agent.executor import DefaultExecutorAgent
from meadow.agent.schema import AgentMessage, Commands
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

DEFAULT_PLANNER_PROMPT = """Based on the following objective provided by the user, please break down the objective into a sequence of sub-steps that one or more agents can solve. For each sub-step in the sequence, indicate which agents should perform the task and generate a detailed instruction for the agent to follow. You can use the same agent multiple times. If you are confused by the task or need more details, please ask for feedback. The user may also provide suggestions to the plan that you should take into account when generating the plan. When generating a plan, please use the following tag format to specify the plan.

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

If the user responds back at some point with a message that indicates the user is satisfied with the plan, ONLY output {termination_message} tags to signal an end to the conversation. {termination_message} tags should only be used in isolation of all other tags.
{serialized_schema}
Below are the agents you have access to.

<agents>
{agents}
</agents>
"""


class SubTaskForParse(BaseModel):
    """Sub-task in a plan used in executor."""

    agent_name: str
    prompt: str


class SubTask:
    """Sub-task in a plan."""

    agent: Agent
    prompt: str

    def __init__(self, agent: Agent, prompt: str):
        self.agent = agent
        self.prompt = prompt


def parse_plan(
    message: str,
    agent_name: str,
    database: Database,
    available_agents: dict[str, Agent],
) -> AgentMessage:
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
    plan: list[SubTaskForParse] = []
    try:
        root = ElementTree.fromstring(inner_steps)  # Parse the XML string
        for step in root:
            agent = (
                step.find("agent").text if step.find("agent") is not None else "Unknown"
            )
            if agent not in available_agents:
                raise ValueError(
                    f"Agent {agent} not found in available agents. Please only use {', '.join(available_agents.keys())}"
                )
            instruction = (
                step.find("instruction").text.strip()
                if step.find("instruction") is not None
                else "No instruction"
            )
            plan.append(SubTaskForParse(agent_name=agent, prompt=instruction))
    except ElementTree.ParseError as e:
        error_message = f"Failed to parse the message as XML. e={e}"
        raise ValueError(error_message)
    return AgentMessage(
        role="assistant",
        content=json.dumps([m.model_dump() for m in plan]),
        display_content=inner_steps,
        tool_calls=None,
        generating_agent=agent_name,
    )


class PlannerAgent(LLMAgent):
    """Agent that generates a plan for a task."""

    def __init__(
        self,
        available_agents: list[Agent],
        client: Client,
        llm_config: LLMConfig,
        database: Database | None,
        executors: list[ExecutorAgent] = None,
        system_prompt: str = DEFAULT_PLANNER_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._available_agents = {a.name: a for a in available_agents}
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._executors = executors
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._plan: Queue[SubTask] = Queue()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

        # Override with defaults
        if self._executors is None:
            self._executors = [
                DefaultExecutorAgent(
                    client=self._client,
                    llm_config=self._llm_config,
                    database=self._database,
                    execution_func=partial(
                        parse_plan, available_agents=self._available_agents
                    ),
                )
            ]

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
            termination_message=Commands.END,
            agents="\n".join(
                [
                    f"<agent>\n{a.name}: {a.description}\n</agent>"
                    for a in self._available_agents.values()
                ]
            ),
        )

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executor agents that should be used by this agent."""
        return self._executors

    def has_plan(self) -> bool:
        """Check if the agent has a plan."""
        return not self._plan.empty()

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
        if self.llm_client is not None:
            print(self.system_message)
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
            print("CONTENT PLANNER", content)
            print("*****")
            if Commands.has_end(content):
                return AgentMessage(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                    generating_agent=self.name,
                    is_termination_message=True,
                )
            else:
                # Try to parse the plan. If it fails, that's okay. The error will be caught by
                # the executor and we will hit the parsing plan option again.
                display_content = None
                try:
                    parsed_plan_message = parse_plan(
                        content, self.name, self._database, self._available_agents
                    )
                    display_content = parsed_plan_message.display_content
                    parsed_plan = [
                        SubTaskForParse(**m)
                        for m in json.loads(parsed_plan_message.content)
                    ]
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
                    pass
                return AgentMessage(
                    role="assistant",
                    content=content,
                    display_content=display_content,
                    generating_agent=self.name,
                )
        else:
            if len(self._available_agents) > 1:
                raise ValueError("No LLM client provided and more than one agent.")
            agent = list(self._available_agents.values())[0]
            raw_content = (
                messages[-1]
                .content.replace("<objective>", "")
                .replace("</objective>", "")
            )
            self._plan.put(SubTask(agent=agent, prompt=raw_content))
            serialized_plan = f"<steps><step1><agent>{agent.name}</agent><instruction>{raw_content}</instruction></step1></steps>"
            return AgentMessage(
                role="assistant",
                content=serialized_plan,
                generating_agent=self.name,
            )
