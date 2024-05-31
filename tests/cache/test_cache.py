import uuid

import pytest

from meadow.cache.cache import Cache
from meadow.client.schema import ChatMessage, ChatRequest, ChatResponse, Choice, Usage


def test_set_and_get_none(duckdb_cache: tuple[Cache, str]) -> None:
    """Test overwriting an existing key's value."""
    cache, _ = duckdb_cache

    with pytest.raises(ValueError):
        cache.set(None, None)

    with pytest.raises(ValueError):
        cache.get(None)


def test_set_and_get(duckdb_cache: tuple[Cache, str]) -> None:
    """Test overwriting an existing key's value."""
    cache, _ = duckdb_cache
    request = ChatRequest(
        messages=[
            {
                "role": "system",
                "content": "Be a bunny",
            },
            {
                "role": "user",
                "content": "What's the weather?",
            },
        ],
        model="gpt-3.5-turbo",
    )
    response = ChatResponse(
        id=str(uuid.uuid4()),
        cached=False,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content="I'm a bunny!"),
            )
        ],
        created=1631987231,
        model="gpt-3.5-turbo",
        usage=Usage(
            completion_tokens=100,
            prompt_tokens=10,
            total_tokens=110,
        ),
    )

    assert cache.get(request) is None

    cache.set(request, response)
    response.cached = True
    assert cache.get(request) == response

    response.created = 1631987232
    cache.set(request, response)
    assert cache.get(request) == response
