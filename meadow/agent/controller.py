"""Controller agent."""

import logging

from meadow.agent.agent import Agent
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, ToolRunner
from meadow.agent.user import UserAgent
from meadow.agent.utils import has_termination_condition, print_message
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

        # In the beginning, the current task agent is the planner so we should
        # generate a plan immediately
        if self._current_task_agent != self._planner:
            assert self._planner.has_plan()

        reply = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )
        if reply.need_user_feedback:
            if reply.is_termination_message:
                return
            await self.send(reply, self._user)
        else:
            await self.send(reply, self._current_task_agent)

    async def generate_tool_call_reply(
        self,
        message: AgentMessage,
    ) -> AgentMessage:
        """Generate a reply based on the tool call."""
        assert message.tool_calls
        assert len(message.tool_calls) == 1, "Only one tool call is supported."
        tool_call = message.tool_calls[0]
        tool_executor = self._tool_executors[tool_call.name]
        try:
            tool_call_response = tool_executor.run(tool_call)
            for_user = True
        except Exception as e:
            tool_call_response = f"Error running {tool_call.name}:\n{e}"
            for_user = False
        return AgentMessage(
            role="assistant",
            content=tool_call_response,
            need_user_feedback=for_user,
            generating_agent=self.name,
        )

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        # We iterate over options to determine how to generate a reply
        # (a) If the last message is a termination message
        #     (i) If the last message is from the user, then we should terminate the conversation
        #     (ii) If the last message is from an agent, then we should move on to next step in task
        #          plan and send to that agent.
        # (b) If the last message is a tool call, run the tool call. If error, send back to agent.
        # otherwise, send to user.
        # (c) Otherwise, the last message is text. If the last message is from the user, then we should
        #     send to the current task agent. If the last message is from an agent, then we should send to user.
        if has_termination_condition(messages[-1].content, self._termination_message):
            if sender == self._user:
                return AgentMessage(
                    role="assistant",
                    content=self._termination_message,
                    generating_agent=self.name,
                    need_user_feedback=True,
                    is_termination_message=True,
                )
            else:
                self._current_task_agent, next_message = (
                    self._planner.move_to_next_agent()
                )
                return AgentMessage(
                    role="assistant", content=next_message, generating_agent=self.name
                )
        elif messages[-1].tool_calls:
            return await self.generate_tool_call_reply(messages[-1])
        else:
            return AgentMessage(
                role="assistant",
                content=messages[-1].content,
                generating_agent=self.name,
                need_user_feedback=(sender != self._user),
            )

    async def initiate_chat(
        self,
        input: str,
        max_plan_steps: int = 1,
    ) -> dict[Agent, list[AgentMessage]]:
        """Chat of user with the agent."""
        # Generate the plan
        step_itr = 0
        while step_itr < max_plan_steps:
            if step_itr == 0:
                message = AgentMessage(
                    role="user", content=input, generating_agent=self._user.name
                )
                await self.receive(message, self._user)
            step_itr += 1
        return self._messages.get_all_messages()
