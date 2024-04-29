"""Class for managing message history between different agents."""

from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage
from meadow.client.schema import Role


class MessageHistory:
    """Class for managing message history between different agents.

    As each agent owns its own history, the key is the other agent. If the
    other agent chats, that the "user". If the owner chats, that's the "assistant".
    """

    def __init__(self) -> None:
        """Initialize the message history."""
        self._history: dict[Agent, list[AgentMessage]] = {}

    def add_message(self, agent: Agent, role: Role, message: AgentMessage) -> None:
        """Add a message to the message history."""
        if agent not in self._history:
            self._history[agent] = []
        # make a copy of the message to avoid modifying the original
        message = message.model_copy()
        message.role = role
        self._history[agent].append(message)

    def get_messages(
        self, recipient: Agent, skip_termination_pairs: bool = True
    ) -> list[AgentMessage]:
        """Get all messages between two agents.

        Some agents are asked to do tasks multiple times throughout the
        conversation. We still need user exists in between but do not want
        to add those to the later tasks.
        """
        if recipient not in self._history:
            return []
        history = self._history[recipient]
        if skip_termination_pairs and len(history) > 1:
            # iterate over user-assistant pairs and remove the user response that trigger termination and the termination message
            history_to_drop = []
            last_msg_idx = 0
            last_msg = history[0]
            for msg in history[1:]:
                if (
                    last_msg.role == "user"
                    and msg.role == "assistant"
                    and msg.is_termination_message
                ):
                    history_to_drop.extend([last_msg_idx, last_msg_idx + 1])
                last_msg_idx += 1
                last_msg = msg
            history = [
                msg for idx, msg in enumerate(history) if idx not in history_to_drop
            ]
        return history

    def get_all_messages(self) -> dict[Agent, list[AgentMessage]]:
        """Get all messages in the history."""
        return self._history
