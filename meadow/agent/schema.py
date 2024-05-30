"""Agent schema classes."""

import enum
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel, model_validator

from meadow.client.schema import ChatMessage
from meadow.database.database import Database

logger = logging.getLogger(__name__)


class Commands:
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
        return Commands._has_signal_string(content, Commands.NEXT)

    @staticmethod
    def has_end(content: str) -> bool:
        return Commands._has_signal_string(content, Commands.END)


class AgentMessage(ChatMessage):
    """The contents of the message."""

    """Content that is for user display only.

    Useful when we want to carry the more detailed content into the chat history but only show the structured output to the user.
    """
    display_content: str | None = None

    sending_agent: str

    receiving_agent: str | None = None

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


class AgentRole(enum.Enum):
    """Agent role."""

    SUPERVISOR = enum.auto()
    EXECUTOR = enum.auto()


class ExecutorFunctionInput:
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
