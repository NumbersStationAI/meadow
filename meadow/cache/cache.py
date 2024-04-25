"""Cache for queries and responses."""

import json
from abc import ABC, abstractmethod

from meadow.client.schema import ChatRequest, ChatResponse


def serialize_request(request: ChatRequest) -> str:
    """Serialize a request."""
    return json.dumps(request.model_dump(exclude_none=True))


def serialize_response(response: ChatResponse) -> str:
    """Serialize a response."""
    return json.dumps(response.model_dump())


class Cache(ABC):
    """A cache for request/response pairs."""

    @abstractmethod
    def close(self) -> None:
        """Close the cache."""
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_key(self, key: str) -> str | None:
        """
        Get the key for a request.

        With return None if key is not in cache.

        Args:
            key: key for cache.
            table: table to get key in.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def set_key(self, key: str, value: str) -> None:
        """
        Set the value for the key.

        Will override old value.

        Args:
            key: key for cache.
            value: new value for key.
            table: table to set key in.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_all_keys(self) -> list[str]:
        """
        Get all keys in cache.

        Returns:
            List of keys in cache.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def commit(self) -> None:
        """Commit any results."""
        raise NotImplementedError()  # pragma: no cover

    def get(self, request: ChatRequest) -> ChatResponse | None:
        """Get the result of request.

        Args:
            request: request to get.

        Returns:
            Response object or None if not in cache.
        """
        if not request:
            raise ValueError("Request is required.")
        key = serialize_request(request)
        cached_response = self.get_key(key)
        if cached_response:
            response = ChatResponse.model_validate(json.loads(cached_response))
            response.cached = True
            return response
        return None

    def set(self, request: ChatRequest, response: ChatResponse) -> None:
        """Set the value for the key.

        Args:
            request: request to set.
            response: response to set.
        """
        if not request:
            raise ValueError("Request is required.")
        key = serialize_request(request)
        self.set_key(key, serialize_response(response))
