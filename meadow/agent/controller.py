"""Controller agent."""

import logging

from meadow.agent.agent import Agent
from meadow.agent.executor import ExecutorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, Commands, ToolRunner
from meadow.agent.user import UserAgent
from meadow.agent.utils import print_message
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


class ControllerAgent(Agent):
    """The controller agent that manages the flow between other agents."""

    def __init__(
        self,
        user: UserAgent,
        planner: PlannerAgent,
        tool_executors: list[ToolRunner] = None,
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
        if planner.executors:
            self._agent_executors[planner] = planner.executors
        # Current agent we are actively talking to (can be use)
        self._current_agent: Agent = self._planner
        # Current agent solving the task (not a user)
        self._current_task_agent: Agent = self._planner
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "Controller"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Overlord"

    def set_current_agent(self, agent: Agent) -> None:
        """Set the current task agent."""
        self._current_agent = agent

    def set_current_task_agent(self, agent: Agent) -> None:
        """Set the current task."""
        self._current_task_agent = agent

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
        await self.send(reply, self._current_agent)

    async def _generate_next_step_reply(self) -> AgentMessage:
        """Generate a reply to move to the next step in the task plan."""
        next_task = self._planner.move_to_next_agent()
        if next_task:
            self.set_current_task_agent(next_task.agent)
            self.set_current_agent(next_task.agent)
            next_message = next_task.prompt
            assert next_message, "Prompt cannot be empty"
        else:
            self.set_current_agent(self._user)
            next_message = Commands.END
        return AgentMessage(
            role="assistant",
            content=next_message,
            generating_agent=self.name,
            is_termination_message=next_message == Commands.END,
        )

    async def _generate_executed_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> AgentMessage:
        """Generate a reply after execution.

        If the execution does not error, return the last executor response which could do some
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
            executor_response = await executor.generate_reply(messages, sender)
            if executor_response.is_error_message:
                self.set_current_agent(sender)
                return executor_response
        final_response = executor_response or default_response
        # The current_task_agent is the agent of the task that is currently answering the user's
        # question. current_agent is whoever we last chatted with. When a user responds, current_agent
        # will be the user and current_task_agent will be the llm agent of the task
        self.set_current_agent(
            self._user if sender != self._user else self._current_task_agent
        )
        return final_response

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        # TODO: add comments once finalized
        if Commands.has_next(messages[-1].content):
            assert sender == self._user, "Only user can send move on message."
            return await self._generate_next_step_reply()
        elif Commands.has_end(messages[-1].content):
            # If user wants to quit, return to them and quit immediately
            # But if a agent says to end, that means move on to next step
            if sender == self._user:
                self.set_current_agent(self._user)
                return AgentMessage(
                    role="assistant",
                    content=Commands.END,
                    generating_agent=self.name,
                    is_termination_message=True,
                )
            else:
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
