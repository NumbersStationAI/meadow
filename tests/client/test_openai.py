from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from meadow.client.api.openai import OpenAIClient
from meadow.client.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice as MeadowChoice,
    ToolCall,
    Usage,
)


@pytest.fixture
def openai_client() -> OpenAIClient:
    with patch("meadow.client.api.openai.AsyncOpenAI") as mock:
        client = OpenAIClient("fake_api")
        mock.chat.completions.create = AsyncMock()
        return client


@pytest.fixture
def chat_completion() -> ChatCompletion:
    """Chat completion response."""
    return ChatCompletion(
        id="chatcmpl-9HJjFKeo5ctacqavGzhx2GfUNkCCX",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="This is a test.",
                    role="assistant",
                    function_call=None,
                ),
            )
        ],
        created=1713914441,
        model="gpt-3.5-turbo-0125",
        object="chat.completion",
        system_fingerprint="fp_c2295e73ad",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=12, total_tokens=17),
    )


@pytest.fixture
def chat_tool_completion() -> ChatCompletion:
    """Chat completion response with tool call."""
    return ChatCompletion(
        id="chatcmpl-9HeeUM5KHzZGMojTc8PBZ6QwKebYg",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_i0wlYpIWQSPXUPkJeKQZCFxs",
                            function=Function(
                                arguments='{"question":"What is the total revenue by product category for the year 2021?"}',
                                name="query_gen",
                            ),
                            type="function",
                        )
                    ],
                ),
            )
        ],
        created=1713994870,
        model="gpt-3.5-turbo-0125",
        object="chat.completion",
        system_fingerprint="fp_c2295e73ad",
        usage=CompletionUsage(
            completion_tokens=28, prompt_tokens=145, total_tokens=173
        ),
    )


def test_init(openai_client: OpenAIClient) -> None:
    """Test initialization of the OpenAI client."""
    assert openai_client.client is not None


def test_convert_request_for_openai(
    openai_client: OpenAIClient, chat_request: ChatRequest
) -> None:
    """Test conversion of ChatRequest to OpenAI format."""
    request_dict = openai_client.convert_request_for_openai(chat_request)
    assert request_dict == {
        "model": "gpt-3.5-turbo",
        "n": 1,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_gen",
                    "description": "Query generator",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "description": "The question to generate a query for",
                                "type": "string",
                            }
                        },
                        "required": ["question"],
                    },
                },
            }
        ],
        "messages": [{"role": "user", "content": "Say this is a test"}],
    }


def test_convert_openai_to_response(
    openai_client: OpenAIClient,
    chat_completion: ChatCompletion,
    chat_tool_completion: ChatCompletion,
) -> None:
    """Test conversion of OpenAI response to ChatResponse."""
    response = openai_client.convert_openai_to_response(chat_completion)
    assert response == ChatResponse(
        id="chatcmpl-9HJjFKeo5ctacqavGzhx2GfUNkCCX",
        cached=False,
        choices=[
            MeadowChoice(
                index=0,
                message=ChatMessage(
                    content="This is a test.",
                    role="assistant",
                ),
            )
        ],
        created=1713914441,
        model="gpt-3.5-turbo-0125",
        usage=Usage(completion_tokens=5, prompt_tokens=12, total_tokens=17),
    )

    response = openai_client.convert_openai_to_response(chat_tool_completion)
    assert response == ChatResponse(
        id="chatcmpl-9HeeUM5KHzZGMojTc8PBZ6QwKebYg",
        cached=False,
        choices=[
            MeadowChoice(
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


@pytest.mark.asyncio
async def test_arun_chat(
    openai_client: OpenAIClient,
    chat_request: ChatRequest,
    chat_completion: ChatCompletion,
) -> None:
    """Test sending a chat request."""
    # make return value work with await expression
    mock_response = AsyncMock(return_value=chat_completion)
    openai_client.client.chat.completions.create = mock_response  # type: ignore
    response = await openai_client.arun_chat(chat_request)
    openai_client.client.chat.completions.create.assert_called_once_with(
        **openai_client.convert_request_for_openai(chat_request)
    )
    assert response == ChatResponse(
        id="chatcmpl-9HJjFKeo5ctacqavGzhx2GfUNkCCX",
        cached=False,
        choices=[
            MeadowChoice(
                index=0,
                message=ChatMessage(
                    content="This is a test.",
                    role="assistant",
                ),
            )
        ],
        created=1713914441,
        model="gpt-3.5-turbo-0125",
        usage=Usage(completion_tokens=5, prompt_tokens=12, total_tokens=17),
    )
