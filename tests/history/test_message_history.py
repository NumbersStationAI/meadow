"""Test for MessageHistory class."""

from unittest.mock import Mock

import pytest

from meadow.agent.agent import Agent
from meadow.agent.schema import ChatMessage
from meadow.history.message_history import MessageHistory


@pytest.fixture
def mock_agent() -> Mock:
    return Mock(spec=Agent)


@pytest.fixture
def mock_message() -> Mock:
    return Mock(spec=ChatMessage)


def test_add_message_empty_history(mock_agent: Mock, mock_message: Mock) -> None:
    """Test adding a message to an empty history."""
    message_history = MessageHistory()
    assert (
        mock_agent not in message_history.get_all_messages()
    )  # Ensure history is initially empty for the agent
    message_history.add_message(mock_agent, mock_message)
    assert message_history.get_messages(mock_agent) == [mock_message]
    # Adding another message
    another_message = Mock(spec=ChatMessage)
    message_history.add_message(mock_agent, another_message)
    assert message_history.get_messages(mock_agent) == [mock_message, another_message]


def test_get_messages_non_existent_agent(mock_agent: Mock) -> None:
    """Test getting messages for a non-existent agent."""
    message_history = MessageHistory()
    assert message_history.get_messages(mock_agent) == []


def test_get_all_messages_multiple_agents() -> None:
    """Test getting all messages with multiple agents."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    agent_two = Mock(spec=Agent)
    message_one = Mock(spec=ChatMessage)
    message_two = Mock(spec=ChatMessage)
    message_three = Mock(spec=ChatMessage)

    message_history.add_message(agent_one, message_one)
    message_history.add_message(agent_two, message_two)
    message_history.add_message(agent_one, message_three)

    expected_history = {
        agent_one: [message_one, message_three],
        agent_two: [message_two],
    }
    assert message_history.get_all_messages() == expected_history
