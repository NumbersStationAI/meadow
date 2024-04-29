"""Agent schema classes."""

import logging
from typing import Any, Callable

from pydantic import BaseModel, model_validator

from meadow.client.schema import ChatMessage, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class AgentMessage(ChatMessage):
    """The contents of the message."""

    """Content that is for display only.

    Useful when we want to carry the more detailed content into the chat history but only show the structured output to the user.
    """
    display_content: str = None

    generating_agent: str

    need_user_feedback: bool = False

    is_termination_message: bool = False

    @model_validator(mode="after")
    def set_default_second_attr(self):
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
