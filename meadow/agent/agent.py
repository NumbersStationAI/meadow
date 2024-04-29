from meadow.agent.schema import AgentMessage
from meadow.client.client import Client


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
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""


class LLMAgent(Agent):
    """LLM agent."""

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""


class DataAgent(LLMAgent):
    """Agent for data tasks."""

    @property
    def table(self) -> str:
        """The table used by the agent."""
        ...
