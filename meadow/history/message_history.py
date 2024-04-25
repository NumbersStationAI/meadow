"""Class for managing message history between different agents."""

from meadow.agent.agent import Agent
from meadow.agent.schema import ChatMessage


class MessageHistory:
    """Class for managing message history between different agents."""

    def __init__(self):
        """Initialize the message history."""
        self._history: dict[Agent, list[ChatMessage]] = {}

    def add_message(self, sender: Agent, message: ChatMessage) -> None:
        """Add a message to the message history.

        Args:
            sender (Agent): the agent that sent the message.
            message (str): the message sent by the sender.
        """
        if sender not in self._history:
            self._history[sender] = []
        self._history[sender].append(message)

    def get_messages(self, sender) -> list[ChatMessage]:
        """Get all messages between two agents.

        Args:
            sender (Agent): the agent that sent the messages.

        Returns:
            list: a list of messages between the two agents.
        """
        if sender in self._history:
            return self._history[sender]
        return []

    def get_all_messages(self) -> dict[Agent, list[ChatMessage]]:
        """Get all messages in the history.

        Returns:
            dict: a dictionary of messages between different agents.
        """
        return self._history
