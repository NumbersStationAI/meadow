"""Class for managing message history between different agents."""

import time

from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage, ClientMessageRole


def is_time_unique(
    history: dict[Agent, list[AgentMessage]], time_to_check: float
) -> bool:
    """Check if the time is unique in the history."""
    for messages in history.values():
        for message in messages:
            if message.creation_time == time_to_check:
                return False
    return True


class MessageHistory:
    """Class for managing message history between different agents.

    As each agent owns its own history, the key is the other agent. If the
    other agent chats, that the "user". If the owner chats, that's the "assistant".
    """

    def __init__(self) -> None:
        """Initialize the message history."""
        self._history: dict[Agent, list[AgentMessage]] = {}

    def add_message(
        self, agent: Agent, agent_role: ClientMessageRole, message: AgentMessage
    ) -> None:
        """Add a message to the message history."""
        if agent not in self._history:
            self._history[agent] = []
        # make a copy of the message to avoid modifying the original
        message = message.model_copy()
        if agent_role == ClientMessageRole.SENDER:
            message.receiving_agent = agent.name
        message.agent_role = agent_role
        # Set the role of the message. This typically happens internally
        message.role = agent_role.value
        # Setting the time on add enforces uniqueness across all history agents
        message.creation_time = time.time()
        assert is_time_unique(self._history, message.creation_time)
        self._history[agent].append(message)

    def copy_messages_from(self, agent: Agent, messages: list[AgentMessage]) -> None:
        """Copy a list of messages to another agent."""
        for message in messages:
            self.add_message(agent, message.agent_role, message)

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
                    last_msg.agent_role == ClientMessageRole.RECEIVER
                    and msg.agent_role == ClientMessageRole.SENDER
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
        to_return = {}
        for agent, history in self._history.items():
            to_return[agent] = history
        return to_return

    def get_messages_linearly_by_time(self) -> list[AgentMessage]:
        """Get all messages in the history linearly by time."""
        all_messages: list[AgentMessage] = []
        for messages in self._history.values():
            all_messages.extend(messages)
        all_messages.sort(key=lambda x: x.creation_time)
        return all_messages
