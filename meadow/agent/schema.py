"""Agent schema classes."""

from datetime import datetime
import logging
from typing import Any, Callable

from pydantic import BaseModel, model_validator

from meadow.client.schema import ChatMessage, ToolCall, ToolSpec

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

    generating_agent: str

    is_termination_message: bool = False

    """Does the message require an explicit response. If so, auto-response
    won't fire."""
    requires_response: bool = False

    """If message requires being sent to the agent's executors."""
    requires_execution: bool = False

    creation_time: datetime = datetime.now()

    @model_validator(mode="after")
    def set_default_second_attr(self) -> "AgentMessage":
        """
        Set second_attr to the value of first_attr if second_attr is not explicitly provided.
        """
        if self.display_content is None:
            self.display_content = self.content
        return self


class ToolRunner(BaseModel):
    """Executor for a tool."""

    """Tool spec sent to model."""
    tool_spec: ToolSpec

    """Default arguments for the tool."""
    default_arguments: dict[str, Any] = {}

    """Executor function."""
    executor: Callable[..., Any]

    def run(self, tool_call: ToolCall) -> Any:
        """Run the tool."""
        all_args = self.default_arguments.copy()
        all_args.update(tool_call.arguments)
        logger.info(f"Running tool={self.tool_spec.name}, args={all_args}")
        return self.executor(**all_args)
