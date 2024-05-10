from typing import Callable

from meadow.agent.schema import AgentMessage
from meadow.client.client import Client
from meadow.database.database import Database


class Agent:
    """Agent interface."""

    @property
    def name(self) -> str:
        """The name of the agent."""
        ...

    @property
    def description(self) -> str:
        """The description of the agent. Used for the agent's introduction in
        a group chat setting."""
        ...

    @property
    def executors(self) -> list["ExecutorAgent"] | None:
        """The executors of the agent."""
        ...

    async def send(
        self,
        message: AgentMessage,
        recipient: "Agent",
    ) -> None:
        """Send a message to another agent."""
        ...

    async def receive(
        self,
        message: AgentMessage,
        sender: "Agent",
    ) -> None:
        """Receive a message from another agent."""

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: "Agent",
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply based on the received messages."""


class LLMAgent(Agent):
    """LLM agent."""

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""


class ExecutorAgent:
    """Execution agent that execute/validates a response given an execution function."""

    @property
    def executor_func(self) -> Callable[[str, str, str, Database], AgentMessage]:
        """The execution function of this agent."""
        ...

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply based on the received messages."""


class DataAgent(LLMAgent):
    """Agent for data tasks."""

    @property
    def database(self) -> Database:
        """The database used by the agent."""
        ...
