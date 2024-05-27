"""Planner agent."""

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

DEFAULT_PLANNER_PROMPT = """Below is the data schema the user is working with.
{serialized_schema}

Based on the following objective provided by the user, please break down the objective into a sequence of sub-steps that one or more agents can solve.

For each sub-step in the sequence, indicate which agents should perform the task and generate a detailed instruction for the agent to follow. If you want to refer to a previous step, use 'stepXX'. If you want the output from a previous step to be used in the input, please use {{stepXX}} in the instruction to use the last output from stepXX. When generating a plan, please use the following tag format to specify the plan.

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

You have access to the following agents.

<agents>
{agents}
</agents>

Out a single plan."""

DEFAULT_PLANNER_PROMPT = """Based on the following objective provided by the user, please break down the objective into a sequence of sub-steps that one or more agents can solve.

For each sub-step in the sequence, indicate which agents should perform the task and generate a detailed instruction for the agent to follow. If you want the output from a previous step to be used in the input, please use {{stepXX}} in the instruction to use the last output from stepXX. When generating a plan, please use the following tag format to specify the plan.

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
    message: str,
    agent_name: str,
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
    if message.endswith("</steps"):
        message += ">"
    if "<steps>" not in message:
        raise ValueError(f"Plan not found in the response. message={message}")
    inner_steps = re.search(r"(<steps>.*</steps>)", message, re.DOTALL).group(1)
    parsed_steps = parse_steps(inner_steps)
    plan: list[SubTaskForParse] = []
    for agent, instruction in parsed_steps:
        if agent not in available_agents:
            raise ValueError(
                f"Agent {agent} not found in available agents. Please only use {', '.join(available_agents.keys())}"
            )
        if agent == "AttributeDetector" and "{" in instruction:
            raise ValueError(
                f"AttributeDetector agent cannot have any replacement tags in the instruction. instruction={instruction}"
            )
        # if agent in {"SQLGenerator", "NestedSQLGenerator"} and "{" in instruction:
        #     copied_plan = [p.model_copy() for p in plan] + [
        #         SubTaskForParse(agent_name=agent, prompt=instruction)
        #     ]
        #     fake_plan = swap_instruction_replacements_with_agent_names(copied_plan)
        #     if "SQLGenerator" or "NestedSQLGenerator" in [
        #         p.agent_name for p in fake_plan
        #     ]:
        #         raise ValueError(
        #             "SQLGenerator cannot take as input the output of another SQLGenerator or NestedSQLGenerator agent. If you wish to refer to their previous steps, please just say 'stepXX'"
        #         )

        if agent == "NestedSQLGenerator" and "NestedSQLGenerator" in [
            p.agent_name for p in plan
        ]:
            raise ValueError(
                "Only one NestedSQLGenerator agent can be used in a plan without any SQLGenerator agents. "
                "This agent decomposes the question for you."
            )
        if agent == "SQLGenerator" and "NestedSQLGenerator" in [
            p.agent_name for p in plan
        ]:
            raise ValueError(
                "SQLGenerator agent cannot be used in a plan with a NestedSQLGenerator agent. "
                "Just use one NestedSQLGenerator agent to decompose the question."
            )
        plan.append(SubTaskForParse(agent_name=agent, prompt=instruction))
    return AgentMessage(
        role="assistant",
        content=json.dumps([m.model_dump() for m in plan]),
        display_content=inner_steps,
        tool_calls=None,
        sending_agent=agent_name,
        requires_response=False,
    )


class PlannerAgent(LLMPlannerAgent):
    """Agent that generates a plan for a task."""

    def __init__(
        self,
        available_agents: list[Agent],
        client: Client | None,
        llm_config: LLMConfig | None,
        database: Database | None,
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
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._plan: Queue[SubTask] = Queue()
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
            print(self.system_message)
            print(messages[-1].content)
            print("CONTENT PLANNER", content)
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
                # TODO: refactor using executors
                messages_copy = messages
                attempt_i = 0
                while attempt_i <= 2:
                    attempt_i += 1
                    try:
                        parsed_plan_message = parse_plan(
                            content, self.name, self._available_agents
                        )
                        break
                    except Exception as e:
                        # Do one LLM reask
                        messages_copy = [m.model_copy() for m in messages_copy]
                        messages_copy.append(
                            AgentMessage(
                                role="assistant",
                                content=content,
                                sending_agent=self.name,
                            )
                        )
                        messages_copy.append(
                            AgentMessage(
                                role="user",
                                content=(str(e) + " Please retry."),
                                sending_agent="User",
                            )
                        )
                        chat_response = await generate_llm_reply(
                            client=self.llm_client,
                            messages=messages_copy,
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
                        print("CONTENT PLANNER REDO", content)
                        print("*****")

                display_content = parsed_plan_message.display_content
                parsed_plan = [
                    SubTaskForParse(**m)
                    for m in json.loads(parsed_plan_message.content)
                ]
                # Handle agent name swaps
                parsed_plan = swap_instruction_replacements_with_agent_names(
                    parsed_plan
                )
                for sub_task in parsed_plan:
                    self._plan.put(
                        SubTask(
                            agent=self._available_agents[sub_task.agent_name],
                            prompt=sub_task.prompt,
                        )
                    )
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
                sending_agent=self.name,
            )
