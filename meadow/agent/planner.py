"""Planner agent."""

import json
import logging
import re
from functools import partial
from queue import Queue
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import (
    Agent,
    ExecutorAgent,
    LLMAgentWithExecutors,
    LLMPlannerAgent,
    SubTask,
)
from meadow.agent.executor.reask import ReaskExecutor
from meadow.agent.schema import (
    AgentMessage,
    AgentRole,
    ClientMessageRole,
    Commands,
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

DEFAULT_PLANNER_PROMPT = """Based on the following objective provided by the user, please break down the objective into a sequence of sub-steps that one or more agents can solve.

For each sub-step in the sequence, indicate which agent should perform the task and generate a detailed instruction for the agent to follow. You can use an agent more than once. If you want the output from a previous step to be used in the input, please use {{stepXX}} in the instruction to use the last output from stepXX. When generating a plan, please use the following tag format to specify the plan.

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

You have access to the following agents.

<agents>
{agents}
</agents>

Below is the data schema the user is working with.
{serialized_schema}
"""


class SubTaskForParse(BaseModel):
    """Sub-task in a plan used in executor."""

    agent_name: str
    prompt: str


def parse_replacements_in_instruction(instruction: str) -> list[str]:
    """Parse the agent name in the instruction."""
    instruction_replacements = re.findall(r"\{(.*?)\}", instruction)
    if len(instruction_replacements) == 0:
        return []
    return instruction_replacements


def swap_instruction_replacements_with_agent_names(
    parsed_plan: list[SubTaskForParse],
) -> list[SubTaskForParse]:
    """Replace {{stepXX}} with {{agentNameXX}} for use in the Controller so it knows what conversation to use."""
    for i, sub_task in enumerate(parsed_plan):
        instruction_replacements = parse_replacements_in_instruction(sub_task.prompt)
        for to_replace in instruction_replacements:
            task_i = re.search(r"\d+", to_replace).group(0)
            if int(task_i) >= len(parsed_plan):
                raise ValueError(
                    f"Step {task_i} is not yet defined for replacement in step {i}."
                )
            parsed_plan[i].prompt = parsed_plan[i].prompt.replace(
                f"{{{to_replace}}}", f"{{{parsed_plan[int(task_i) - 1].agent_name}}}"
            )
    return parsed_plan


def parse_steps(input_str: str) -> list[tuple[str, str]]:
    """
    Parse the given XML-like string and extract agent, instruction pairs using regular expressions.
    """
    # Use '.*' after last instruction because sometimes it'll add extra pieces we don't care about
    pattern = re.compile(
        r"<step\d*>\s*<agent>(.*?)</agent>\s*<instruction>(.*?)</instruction>.*?</step\d*>",
        re.DOTALL,
    )
    matches = pattern.findall(input_str)
    return [(agent.strip(), instruction.strip()) for agent, instruction in matches]


def parse_plan(
    input: ExecutorFunctionInput,
    available_agents: dict[str, Agent],
    constraints: list[Callable[[list[SubTask], str], str | None]],
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
    error_message = None
    message = input.messages[-1].content
    userinput = re.search(r"<userinput>(.*?)</userinput>", message, re.DOTALL).group(1)
    if message.endswith("</steps"):
        message += ">"
    if "<steps>" not in message:
        error_message = "Plan not found in the response. Please provide a plan in the format <steps>...</steps>."
    if "</steps>" not in message:
        message += "</steps>"
    if not error_message:
        inner_steps = re.search(r"(<steps>.*</steps>)", message, re.DOTALL).group(1)
        parsed_steps = parse_steps(inner_steps)
        for agent, instruction in parsed_steps:
            if agent not in available_agents:
                error_message = f"Agent {agent} not found in available agents. Please only use {', '.join(available_agents.keys())}."
                break
    if not error_message:
        parsed_plan: list[SubTaskForParse] = []
        plan: list[SubTask] = []
        for agent, instruction in parsed_steps:
            parsed_plan.append(SubTaskForParse(agent_name=agent, prompt=instruction))
            plan.append(SubTask(agent=available_agents[agent], prompt=instruction))
        for constraint in constraints:
            error_message = constraint(plan, userinput)
            if error_message:
                break
    if not error_message:
        try:
            parsed_plan = swap_instruction_replacements_with_agent_names(parsed_plan)
        except Exception as e:
            error_message = (
                f"Error swapping instruction replacements with agent names. e={e}"
            )
    if error_message:
        if input.can_reask_again:
            return AgentMessage(
                content=error_message.strip() + " Please retry.",
                requires_response=True,
                sending_agent=input.agent_name,
            )
        else:
            return AgentMessage(
                content=f"Current plan.\n\n{message}\n\nWe're having trouble generating a plan. Please try to rephrase.",
                sending_agent=input.agent_name,
            )
    return AgentMessage(
        content=json.dumps([m.model_dump() for m in parsed_plan]),
        display_content=inner_steps,
        sending_agent=input.agent_name,
    )


class PlannerAgent(LLMPlannerAgent, LLMAgentWithExecutors):
    """Agent that generates a plan for a task."""

    def __init__(
        self,
        available_agents: list[Agent],
        client: Client | None,
        llm_config: LLMConfig | None,
        database: Database | None,
        executors: list[ExecutorAgent] = None,
        constraints: list[Callable[[list[SubTask], str], str | None]] = [],
        name: str = "Planner",
        description: str = "Generates a plan for a task.",
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
        self._constraints = constraints
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._role = AgentRole.TASK_HANDLER
        self._plan: Queue[SubTask] = Queue()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

        if self._executors is None:
            self._executors = [
                ReaskExecutor(
                    client=None,
                    llm_config=None,
                    database=self._database,
                    execution_func=partial(
                        parse_plan,
                        available_agents=self.available_agents,
                        constraints=self.plan_constraints,
                    ),
                    # More execution attempts for more constraints
                    max_execution_attempts=max(
                        int(len(self._constraints or []) * 1.3), 2
                    ),
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
    def system_message(self) -> str:
        """Get the system message."""
        if self._database is not None:
            serialized_schema = serialize_as_list(self._database.tables)
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

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent."""
        self._role = role

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executor agents that should be used by this agent."""
        return self._executors

    @property
    def available_agents(self) -> dict[str, Agent]:
        """Get the available agents."""
        return self._available_agents

    @property
    def plan_constraints(self) -> list[Callable[[list[SubTask], str], str | None]]:
        """Get the constraints for the planner."""
        return self._constraints

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
        """Generate a reply based on the received messages."""
        if self.llm_client is not None:
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
            # Add back user input for use in the plan parsing constraints
            if "<userinput>" not in content:
                content = f"<userinput>{messages[0].content}</userinput>\n{content}"
            if Commands.has_end(content):
                return AgentMessage(
                    content=content,
                    sending_agent=self.name,
                    is_termination_message=True,
                )
            else:
                self._plan = Queue()
                # parse_plan will check and rethrow any errors in the executor phase. However, we
                # need to update the planner state with the output of the parse_plan function.
                # So we call it here as well. If there's an error, that's okay, but planner
                # agent will get recalled by the validator to fix it.
                parsed_plan_message = parse_plan(
                    input=ExecutorFunctionInput(
                        messages=[
                            AgentMessage(
                                content=content,
                                sending_agent=self.name,
                            )
                        ],
                        agent_name=self.name,
                        database=self._database,
                        can_reask_again=True,
                    ),
                    available_agents=self.available_agents,
                    constraints=self.plan_constraints,
                )
                try:
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
                except Exception:
                    pass
                return AgentMessage(
                    content=content,
                    sending_agent=self.name,
                    requires_execution=True,
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
                content=serialized_plan,
                sending_agent=self.name,
            )
