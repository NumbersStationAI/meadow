from abc import ABC, abstractmethod

from meadow.client.schema import ChatRequest, ChatResponse


class APIClient(ABC):
    """Model client class."""

    @abstractmethod
    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        raise NotImplementedError()  # pragma: no cover
