import logging

from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage
from meadow.agent.utils import print_message
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


class UserAgent(Agent):
    """User agent.

    This agent is a wrapper around user inputs.
    """

    def __init__(
        self,
        name: str,
        description: str = None,
        silent: bool = True,
    ):
        self._name = name
        self._messages = MessageHistory()
        self._description = description
        self._silent = silent

    @property
    def name(self) -> str:
        """The name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """The description of the agent. Used for the agent's introduction in
        a group chat setting."""
        return self._description

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
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
        reply = await self.generate_reply(messages=[message], sender=sender)
        await self.send(reply, sender)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        # Get the input from a user
        reply = input(">>> ")
        return AgentMessage(role="assistant", content=reply, generating_agent=self.name)
