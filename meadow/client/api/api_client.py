from abc import ABC, abstractmethod

from meadow.client.schema import Request, Response


class APIClient(ABC):
    """Model client class."""

    @abstractmethod
    async def arun_request(self, request: Request) -> Response:
        """Send a chat request."""
        raise NotImplementedError()
