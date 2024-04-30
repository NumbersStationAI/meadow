"""Test for MessageHistory class."""

from unittest.mock import Mock

import pytest

from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage
from meadow.history.message_history import MessageHistory


@pytest.fixture
def mock_agent() -> Mock:
    return Mock(spec=Agent)


def test_add_message(mock_agent: Mock) -> None:
    """Test adding a message to an empty history."""
    message_history = MessageHistory()
    assert (
        mock_agent not in message_history.get_all_messages()
    )  # Ensure history is initially empty for the agent

    message = AgentMessage(
        content="Hello, world!", generating_agent="test", role="user"
    )
    message_history.add_message(mock_agent, "user", message)
    assert message_history.get_messages(mock_agent) == [message]
    # Adding another message
    another_message = AgentMessage(
        content="Bye, world!", generating_agent="test", role="assistant"
    )
    message_history.add_message(mock_agent, "assistant", another_message)
    assert message_history.get_messages(mock_agent) == [message, another_message]
    # Add third message and switch roles
    third_message = AgentMessage(
        content="Hello again!", generating_agent="test", role="user"
    )
    message_history.add_message(mock_agent, "assistant", third_message)
    assert message_history.get_messages(mock_agent)[-1].role == "assistant"
    assert message_history.get_messages(mock_agent)[-1].content == "Hello again!"


def test_get_messages_non_existent_agent(mock_agent: Mock) -> None:
    """Test getting messages for a non-existent agent."""
    message_history = MessageHistory()
    assert message_history.get_messages(mock_agent) == []


def test_get_all_messages_multiple_agents() -> None:
    """Test getting all messages with multiple agents."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    agent_two = Mock(spec=Agent)
    message_one = AgentMessage(
        content="Hello, world!", role="assistant", generating_agent="agent_one"
    )
    message_two = AgentMessage(
        content="Hello, world 2!", role="assistant", generating_agent="agent_two"
    )
    message_three = AgentMessage(
        content="Hello, world 3!", role="assistant", generating_agent="agent_one"
    )

    message_history.add_message(agent_one, "assistant", message_one)
    message_history.add_message(agent_two, "assistant", message_two)
    message_history.add_message(agent_one, "assistant", message_three)

    expected_history = {
        agent_one: [message_one, message_three],
        agent_two: [message_two],
    }
    assert message_history.get_all_messages() == expected_history


def test_get_messages_skip_exit() -> None:
    """Test getting all messages with multiple agents."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    message_one = AgentMessage(
        content="Hello, world!", role="assistant", generating_agent="agent_one"
    )
    message_two = AgentMessage(content="I'm done", role="user", generating_agent="user")
    message_three = AgentMessage(
        content="See ya <exit>",
        role="assistant",
        generating_agent="agent_one",
        is_termination_message=True,
    )
    message_four = AgentMessage(
        content="Show me cats", role="user", generating_agent="user"
    )

    message_history.add_message(agent_one, "assistant", message_one)
    message_history.add_message(agent_one, "user", message_two)
    message_history.add_message(agent_one, "assistant", message_three)
    message_history.add_message(agent_one, "user", message_four)

    assert message_history.get_messages(agent_one) == [message_one, message_four]
