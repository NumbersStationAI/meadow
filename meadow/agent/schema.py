"""Agent schema classes."""

import enum
import logging
import time

from pydantic import model_validator

from meadow.client.schema import ChatMessage
from meadow.database.database import Database

logger = logging.getLogger(__name__)


class Commands:
    """DSL commands interpreted by the controller."""

    NEXT = "<next>"
    END = "<end>"

    @staticmethod
    def _has_signal_string(content: str, signal_str: str) -> bool:
        """Check if the message contains signalling string."""
        return content.strip().endswith(signal_str) or content.strip().startswith(
            signal_str
        )

    @staticmethod
    def has_next(content: str) -> bool:
        """Check if the message contains the next signal."""
        return Commands._has_signal_string(content, Commands.NEXT)

    @staticmethod
    def has_end(content: str) -> bool:
        """Check if the message contains the end signal."""
        return Commands._has_signal_string(content, Commands.END)


class ClientMessageRole(str, enum.Enum):
    """The role of the author of the message.

    LLMs in chat messages require the role to be "user", "assistant", or "system".
    In the agent chat framework, the role is determined by if an agent is sending
    or receiving a message. We handle that mapping here.

    SENDER == "assistant" and RECEIVER == "user".
    """

    SENDER = "assistant"
    RECEIVER = "user"

    # Used internally for agents to define their own system messages for the LLM.
    SYSTEM = "system"


class AgentRole(enum.Enum):
    """Agent role.

    In conversations, an agent can either be a supervisor (i.e. the instigator
    of the conversation) or the task handler (i.e the responder). Some agents serve both
    and their role determines how they respond.
    """

    SUPERVISOR = enum.auto()
    TASK_HANDLER = enum.auto()


class AgentMessage(ChatMessage):
    """The contents of the message."""

    """Content that is for user display only.

    Useful when we want to carry the more detailed content into the chat history but only show the structured output to the user.
    """
    display_content: str | None = None

    sending_agent: str

    receiving_agent: str | None = None

    """The agent role of this message."""
    agent_role: ClientMessageRole = ClientMessageRole.SENDER

    is_termination_message: bool = False

    """Does the message require an explicit response. If so, auto-response
    won't fire."""
    requires_response: bool = False

    """If message requires being sent to the agent's executors."""
    requires_execution: bool = False

    creation_time: float = time.time()

    @model_validator(mode="after")
    def set_default_second_attr(self) -> "AgentMessage":
        """
        Set second_attr to the value of first_attr if second_attr is not explicitly provided.
        """
        if self.display_content is None:
            self.display_content = self.content
        return self

    @model_validator(mode="after")
    def set_chat_role(self) -> None:
        """
        Set chat message role from the agent_role.
        """
        self.role = self.agent_role.value


class ExecutorFunctionInput:
    """Input for an executor function."""

    def __init__(
        self,
        messages: list[AgentMessage],
        agent_name: str,
        database: Database,
        # Can the error be send to the model for correction?
        # If set to False, this means the error will not be fixed."""
        can_reask_again: bool,
    ):
        self.messages = messages
        self.agent_name = agent_name
        self.database = database
        self.can_reask_again = can_reask_again
