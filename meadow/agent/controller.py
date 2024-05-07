"""Controller agent."""

import logging

from meadow.agent.agent import Agent
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, ToolRunner
from meadow.agent.user import UserAgent
from meadow.agent.utils import Commands, print_message
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
        silent: bool = True,
    ):
        self._user = user
        self._planner = planner
        self._messages = MessageHistory()
        self._tool_executors = {t.tool_spec.name: t for t in tool_executors or []}
        self._current_task_agent: Agent = self._planner
        self._termination_message = termination_message
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

        reply = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )

        if reply.is_termination_message:
            return

        await self.send(reply, self._current_task_agent)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        if Commands.has_next(messages[-1].content):
            task = self._planner.get_next_task()
            # If the planner has no more steps, then we should terminate the conversation
            if self._current_task_agent is None:
                self._current_task_agent = self._user
                return AgentMessage(
                    role="assistant",
                    content=self._termination_message,
                    generating_agent=self.name,
                    need_user_feedback=True,
                    is_termination_message=True,
                )
            self._current_task_agent = task.agent
            return AgentMessage(
                role="assistant", content=task.prompt, generating_agent=self.name
            )
        elif Commands.has_end(messages[-1].content):
            self._current_task_agent = self._user
            return AgentMessage(
                role="assistant",
                content=self._termination_message,
                generating_agent=self.name,
                need_user_feedback=True,
                is_termination_message=True,
            )

        """Generate a reply based on the received messages."""
        self._current_task_agent = (
            self._user if sender != self._user else self._current_task_agent
        )
        return AgentMessage(
            role="assistant",
            content=messages[-1].content,
            display_content=messages[-1].display_content,
            generating_agent=self.name,
            need_user_feedback=(sender != self._user),
        )

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
