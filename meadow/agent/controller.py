"""Controller agent."""

import logging

from meadow.agent.agent import Agent
from meadow.agent.executor import ExecutorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, ToolRunner
from meadow.agent.user import UserAgent
from meadow.agent.utils import has_signal_string, print_message
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


class ControllerAgent(Agent):
    """The controller agent that manages the flow between other agents."""

    def __init__(
        self,
        user: UserAgent,
        planner: PlannerAgent,
        tool_executors: list[ToolRunner] = None,
        termination_message: str = "<exit>",
        move_on_message: str = "<next>",
        silent: bool = True,
    ):
        self._user = user
        self._planner = planner
        self._messages = MessageHistory()
        self._tool_executors = {t.tool_spec.name: t for t in tool_executors or []}
        self._agent_executors: dict[Agent, list[ExecutorAgent]] = {
            agent: agent.executors
            for agent in planner._available_agents.values()
            if agent.executors
        }
        self._current_task_agent: Agent = self._planner
        self._termination_message = termination_message
        self._move_on_message = move_on_message
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "Controller"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Overlord"

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
        self._messages.add_message(agent=sender, role="user", message=message)

        # In the beginning, the current task agent is the planner so we should
        # generate a plan immediately
        if self._current_task_agent != self._planner:
            assert self._planner.has_plan()

        reply, to_send = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )
        if reply.is_termination_message:
            return
        await self.send(reply, to_send)

    async def _generate_next_step_reply(self) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply to move to the next step in the task plan."""
        self._current_task_agent, next_message = self._planner.move_to_next_agent()
        # If the planner has no more steps, then we should terminate the conversation
        if self._current_task_agent is None:
            return AgentMessage(
                role="assistant",
                content=self._termination_message,
                generating_agent=self.name,
                is_termination_message=True,
            ), self._user
        return AgentMessage(
            role="assistant", content=next_message, generating_agent=self.name
        ), self._current_task_agent

    async def _generate_executed_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply after validation.

        If the validation is never an error, return the last executor response which could do some
        parsing of the output.

        Otherwise, return the first error message.
        """
        executor_response = None
        default_response = AgentMessage(
            role="assistant",
            content=messages[-1].content,
            display_content=messages[-1].display_content,
            generating_agent=self.name,
        )
        for executor in self._agent_executors.get(sender, []):
            executor_response, _ = await executor.generate_reply(messages, sender)
            if executor_response.is_error_message:
                return executor_response, sender
        # We know that executor_response is not an error message
        final_response = executor_response or default_response
        return (
            final_response,
            self._user if sender != self._user else self._current_task_agent,
        )

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply based on the received messages."""
        # We iterate over options to determine how to generate a reply
        # (a) If the last message is a termination message
        #     (i) If the last message is from the user, then we should terminate the conversation
        #     (ii) If the last message is from an agent, then we should move on to next step in task
        #          plan and send to that agent.
        # (b) If the last message is a DSL "next" message
        #     (i) The last message must be from the user, then we should move on to the next step in the task plan
        # (c) If the last message is a tool call, run the tool call. If error, send back to agent.
        # otherwise, send to user.
        # (d) Otherwise, the last message is text. If the last message is from the user, then we should
        #     send to the current task agent. If the last message is from an agent, then we should send to user.
        if has_signal_string(messages[-1].content, self._termination_message):
            if sender == self._user:
                return AgentMessage(
                    role="assistant",
                    content=self._termination_message,
                    generating_agent=self.name,
                    is_termination_message=True,
                ), self._user
            else:
                return await self._generate_next_step_reply()
        elif has_signal_string(messages[-1].content, self._move_on_message):
            assert sender == self._user, "Only user can send move on message."
            return await self._generate_next_step_reply()
        else:
            return await self._generate_executed_reply(messages, sender)

    async def initiate_chat(
        self,
        input: str,
        max_plan_steps: int = 1,
    ) -> dict[Agent, list[AgentMessage]]:
        """Chat of user with the agent."""
        step_itr = 0
        while step_itr < max_plan_steps:
            if step_itr == 0:
                message = AgentMessage(
                    role="user", content=input, generating_agent=self._user.name
                )
                await self.receive(message, self._user)
            step_itr += 1
        return self._messages.get_all_messages()
