from __future__ import annotations

from abc import abstractmethod
from typing import Callable

from meadow.agent.schema import AgentMessage, AgentRole, ExecutorFunctionInput
from meadow.client.client import Client


class Agent:
    """Agent interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the agent."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """The description of the agent. Used in planners."""
        ...

    @property
    def planner(self) -> LLMPlannerAgent:
        """The planner of the agent."""
        return None

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executors of the agent."""
        return None

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent.

        Only used for agents that have executors."""
        return

    @abstractmethod
    def get_messages(self, chat_agent: Agent) -> list[AgentMessage]:
        """Get the messages between self and the chat_agent."""
        ...

    @abstractmethod
    def add_to_messages(self, chat_agent: Agent, messages: list[AgentMessage]) -> None:
        """Add chat messages between self and chat_agent.

        Used when starting hierarchical chats and historical messages
        need to be passed to the agent.
        """
        ...

    @abstractmethod
    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        ...

    @abstractmethod
    async def receive(
        self,
        message: AgentMessage,
        sender: Agent,
    ) -> None:
        """Receive a message from another agent."""

    @abstractmethod
    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""


class LLMAgent(Agent):
    """LLM agent."""

    @property
    @abstractmethod
    def llm_client(self) -> Client:
        """The LLM client of this agent."""


class LLMAgentWithExecutors(LLMAgent):
    """LLM agent with executors."""

    @property
    def executors(self) -> list[ExecutorAgent] | None:
        """The executors of the agent."""
        raise NotImplementedError

    def set_chat_role(self, role: AgentRole) -> None:
        """Set the chat role of the agent.

        Only used for agents that have executors."""
        raise NotImplementedError


class SubTask:
    """Sub-task in a plan."""

    agent: Agent
    prompt: str

    def __init__(self, agent: Agent, prompt: str):
        self.agent = agent
        self.prompt = prompt


class LLMPlannerAgent(LLMAgent):
    """Agent that makes plan."""

    @property
    @abstractmethod
    def available_agents(self) -> dict[str, Agent]:
        """Get the available agents."""
        raise NotImplementedError

    @property
    @abstractmethod
    def plan_constraints(self) -> list[Callable[[list[SubTask], str], str | None]]:
        """Get the plan constraints."""
        raise NotImplementedError

    @abstractmethod
    def move_to_next_agent(self) -> "SubTask":
        """Move to the next agent in the task plan."""
        raise NotImplementedError


class ExecutorAgent(LLMAgent):
    """Execution agent that execute/validates a response given an execution function."""

    @property
    @abstractmethod
    def execution_func(
        self,
    ) -> Callable[[ExecutorFunctionInput], AgentMessage]:
        """The execution function of this agent."""
        ...

    @abstractmethod
    def reset_execution_attempts(self) -> None:
        """Reset the number of execution attempts."""
        ...
