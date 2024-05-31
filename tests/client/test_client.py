from unittest.mock import AsyncMock, MagicMock

import pytest

from meadow.cache.cache import Cache
from meadow.client.api.api_client import APIClient
from meadow.client.client import Client
from meadow.client.schema import ChatMessage, ChatResponse, Choice, ToolCall, Usage


@pytest.fixture
def chat_response() -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-9HeeUM5KHzZGMojTc8PBZ6QwKebYg",
        cached=False,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            unparsed_arguments='{"question":"What is the total revenue by product category for the year 2021?"}',
                            name="query_gen",
                        )
                    ],
                ),
            )
        ],
        created=1713994870,
        model="gpt-3.5-turbo-0125",
        usage=Usage(completion_tokens=28, prompt_tokens=145, total_tokens=173),
    )


@pytest.fixture
def api_client(chat_response: ChatResponse) -> APIClient:
    client = MagicMock(spec=APIClient)
    client.arun_chat = AsyncMock()
    client.arun_chat.return_value = chat_response
    return client


def test_init_without_client() -> None:
    """Test initializing the client without an API client."""
    with pytest.raises(ValueError):
        Client(api_client=None)


@pytest.mark.asyncio
async def test_without_prompt(api_client: APIClient) -> None:
    """Test fails with None prompts."""
    client = Client(api_client=api_client)
    with pytest.raises(ValueError):
        await client.chat(messages=None)


@pytest.mark.asyncio
async def test_without_model(api_client: APIClient) -> None:
    """Test fails with None prompts."""
    client = Client(api_client=api_client)
    with pytest.raises(ValueError):
        await client.chat(messages=[{"text": "Hello"}])


@pytest.mark.asyncio
async def test_chat_with_cache_hit(
    api_client: APIClient,
    duckdb_cache: tuple[Cache, str],
    chat_response: ChatResponse,
) -> None:
    cache, _ = duckdb_cache
    client = Client(api_client=api_client, cache=cache, model="test_model")
    chat_messages = [{"role": "user", "content": "Say this is a test"}]

    assert len(cache.get_all_keys()) == 0
    assert api_client.arun_chat.call_count == 0  # type: ignore
    response = await client.chat(messages=chat_messages)
    assert response == chat_response
    assert len(cache.get_all_keys()) == 1
    assert api_client.arun_chat.call_count == 1  # type: ignore

    response = await client.chat(messages=chat_messages)
    chat_response.cached = True
    assert response == chat_response
    assert len(cache.get_all_keys()) == 1
    # assert it is still just called once
    assert api_client.arun_chat.call_count == 1  # type: ignore
