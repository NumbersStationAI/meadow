"""Test for MessageHistory class."""

from unittest.mock import Mock

import pytest

from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage, ClientMessageRole
from meadow.history.message_history import MessageHistory


@pytest.fixture
def mock_agent() -> Mock:
    agent = Mock(spec=Agent)
    agent.name = "mock_agent"
    return agent


def assert_message_lists_equal(
    list_one: list[AgentMessage], list_two: list[AgentMessage]
) -> None:
    """Assert that two lists of messages are equal."""
    assert len(list_one) == len(list_two)
    for message_one, message_two in zip(list_one, list_two):
        message_one_dump = message_one.model_dump(exclude=set(["creation_time"]))
        message_two_dump = message_two.model_dump(exclude=set(["creation_time"]))
        assert message_one_dump == message_two_dump


def test_add_message(mock_agent: Mock) -> None:
    """Test adding a message to an empty history."""
    message_history = MessageHistory()
    assert (
        mock_agent not in message_history.get_all_messages()
    )  # Ensure history is initially empty for the agent

    message = AgentMessage(
        content="Hello, world!",
        sending_agent="test",
        agent_role=ClientMessageRole.RECEIVER,
    )
    message_history.add_message(mock_agent, ClientMessageRole.RECEIVER, message)
    assert_message_lists_equal(message_history.get_messages(mock_agent), [message])
    # Adding another message
    another_message = AgentMessage(
        content="Bye, world!",
        sending_agent="test",
        receiving_agent=mock_agent.name,
        agent_role=ClientMessageRole.SENDER,
    )
    message_history.add_message(mock_agent, ClientMessageRole.SENDER, another_message)
    assert_message_lists_equal(
        message_history.get_messages(mock_agent), [message, another_message]
    )
    # Add third message and switch roles
    third_message = AgentMessage(
        content="Hello again!",
        sending_agent="test",
        agent_role=ClientMessageRole.RECEIVER,
    )
    message_history.add_message(mock_agent, ClientMessageRole.SENDER, third_message)
    assert message_history.get_messages(mock_agent)[-1].agent_role.value == "assistant"
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
    agent_one.name = "agent_one"
    agent_two = Mock(spec=Agent)
    agent_two.name = "agent_two"
    message_one = AgentMessage(
        content="Hello, world!",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
    )
    message_two = AgentMessage(
        content="Hello, world 2!",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_two",
        receiving_agent=agent_two.name,
    )
    message_three = AgentMessage(
        content="Hello, world 3!",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
    )

    message_history.add_message(agent_one, ClientMessageRole.SENDER, message_one)
    message_history.add_message(agent_two, ClientMessageRole.SENDER, message_two)
    message_history.add_message(agent_one, ClientMessageRole.SENDER, message_three)

    expected_history = {
        agent_one: [message_one, message_three],
        agent_two: [message_two],
    }
    actual_history = message_history.get_all_messages()
    assert len(actual_history) == len(expected_history)
    for agent, messages in actual_history.items():
        assert_message_lists_equal(messages, expected_history[agent])  # type: ignore


def test_get_messages_skip_exit() -> None:
    """Test getting all messages with multiple agents."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    agent_one.name = "agent_one"
    message_one = AgentMessage(
        content="Hello, world!",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
    )
    message_two = AgentMessage(
        content="I'm done",
        agent_role=ClientMessageRole.RECEIVER,
        sending_agent="user",
        receiving_agent=agent_one.name,
    )
    message_three = AgentMessage(
        content="See ya <exit>",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
        is_termination_message=True,
    )
    message_four = AgentMessage(
        content="Show me cats",
        agent_role=ClientMessageRole.RECEIVER,
        sending_agent="user",
        receiving_agent=agent_one.name,
    )

    message_history.add_message(agent_one, ClientMessageRole.SENDER, message_one)
    message_history.add_message(agent_one, ClientMessageRole.RECEIVER, message_two)
    message_history.add_message(agent_one, ClientMessageRole.SENDER, message_three)
    message_history.add_message(agent_one, ClientMessageRole.RECEIVER, message_four)

    assert_message_lists_equal(
        message_history.get_messages(agent_one), [message_one, message_four]
    )


def test_copy_messages_from() -> None:
    """Test copying messages from one agent to another."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    agent_one.name = "agent_one"
    agent_two = Mock(spec=Agent)
    agent_two.name = "agent_two"
    message_one = AgentMessage(
        content="Hello, world!",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
    )
    message_two = AgentMessage(
        content="Bye, world!",
        agent_role=ClientMessageRole.RECEIVER,
        sending_agent="agent_one",
        receiving_agent=agent_one.name,
    )

    message_history.add_message(agent_one, ClientMessageRole.SENDER, message_one)
    message_history.add_message(agent_one, ClientMessageRole.RECEIVER, message_two)
    message_history.copy_messages_from(agent_two, [message_one, message_two])

    message_one.receiving_agent = agent_two.name
    assert_message_lists_equal(
        message_history.get_messages(agent_two),
        [message_one, message_two],
    )


def test_get_messages_linearly_by_time() -> None:
    """Test getting all messages linearly by time."""
    message_history = MessageHistory()
    agent_one = Mock(spec=Agent)
    agent_two = Mock(spec=Agent)

    message = AgentMessage(
        content="First message",
        agent_role=ClientMessageRole.SENDER,
        sending_agent="agent_one",
    )

    # The times get bumped in add_message
    message_history.add_message(agent_one, ClientMessageRole.SENDER, message)
    message.content = "Second message"
    message_history.add_message(agent_two, ClientMessageRole.RECEIVER, message)
    message.content = "Third message"
    message_history.add_message(agent_one, ClientMessageRole.SENDER, message)

    linear_messages = message_history.get_messages_linearly_by_time()
    assert len(linear_messages) == 3
    assert linear_messages[0].content == "First message"
    assert linear_messages[1].content == "Second message"
    assert linear_messages[2].content == "Third message"
