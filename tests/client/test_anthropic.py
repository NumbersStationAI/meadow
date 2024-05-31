from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types.beta.tools.tool_use_block import ToolUseBlock
from anthropic.types.beta.tools.tools_beta_message import ToolsBetaMessage
from anthropic.types.text_block import TextBlock
from anthropic.types.usage import Usage

from meadow.client.api.anthropic import AnthropicClient
from meadow.client.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice as MeadowChoice,
    FunctionArgSpec,
    ToolCall,
    ToolSpec,
    Usage as MeadowUsage,
)


@pytest.fixture
def anthropic_client() -> AnthropicClient:
    with patch("meadow.client.api.anthropic.AsyncAnthropic") as mock:
        client = AnthropicClient("fake_api")
        mock.beta.tools.messages.create = AsyncMock()
        return client


@pytest.fixture
def chat_completion() -> ToolsBetaMessage:
    """Chat completion response."""
    return ToolsBetaMessage(
        id="msg_01Tmqpke9uViGL1uYuoa2H2C",
        content=[
            TextBlock(
                text='<thinking>\nThe user has directly provided a question to ask the query_gen tool:\n"How many cats do I have?"\n\nSince the query_gen tool takes a single required parameter "question" of type string, and the user has provided the question string, we have all the required information to call the tool.\n</thinking>',
                type="text",
            ),
            ToolUseBlock(
                id="toolu_01ABVvieAV3kJWmwVaPAdwpz",
                input={"question": "How many cats do I have?"},
                name="query_gen",
                type="tool_use",
            ),
        ],
        model="claude-3-opus-20240229",
        role="assistant",
        stop_reason="max_tokens",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=495, output_tokens=100),
    )


def test_init(anthropic_client: AnthropicClient) -> None:
    """Test initialization of the Anthropic client."""
    assert anthropic_client.client is not None


def test_convert_request_for_anthropic(
    anthropic_client: AnthropicClient, chat_request: ChatRequest
) -> None:
    """Test conversion of ChatRequest to Anthropic format."""
    request_dict = anthropic_client.convert_request_for_anthropic(chat_request)
    assert request_dict == {
        "messages": [{"role": "user", "content": "Say this is a test"}],
        "model": "gpt-3.5-turbo",
        "tools": [
            {
                "name": "query_gen",
                "description": "Query generator",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "description": "The question to generate a query for",
                            "type": "string",
                        }
                    },
                    "required": ["question"],
                },
            }
        ],
        "max_tokens": 100,
    }

    # add renamed and removed key
    chat_request.stop = ["stop"]
    chat_request.frequency_penalty = 0.5
    request_dict = anthropic_client.convert_request_for_anthropic(chat_request)
    assert request_dict == {
        "messages": [{"role": "user", "content": "Say this is a test"}],
        "model": "gpt-3.5-turbo",
        "stop_sequences": ["stop"],
        "tools": [
            {
                "name": "query_gen",
                "description": "Query generator",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "description": "The question to generate a query for",
                            "type": "string",
                        }
                    },
                    "required": ["question"],
                },
            }
        ],
        "max_tokens": 100,
    }


def test_convert_anthropic_to_response(
    anthropic_client: AnthropicClient,
    chat_completion: ToolsBetaMessage,
) -> None:
    """Test conversion of Anthropic response to ChatResponse."""
    response = anthropic_client.convert_anthropic_to_response(chat_completion)
    assert (
        response
        == ChatResponse(
            id="msg_01Tmqpke9uViGL1uYuoa2H2C",
            cached=False,
            choices=[
                MeadowChoice(
                    index=0,
                    message=ChatMessage(
                        content='<thinking>\nThe user has directly provided a question to ask the query_gen tool:\n"How many cats do I have?"\n\nSince the query_gen tool takes a single required parameter "question" of type string, and the user has provided the question string, we have all the required information to call the tool.\n</thinking>',
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                unparsed_arguments='{"question": "How many cats do I have?"}',
                                name="query_gen",
                            )
                        ],
                    ),
                )
            ],
            created=response.created,  # hack to match the timestamps as we add time in the conversion
            model="claude-3-opus-20240229",
            usage=MeadowUsage(
                completion_tokens=100, prompt_tokens=495, total_tokens=595
            ),
        )
    )


def test_convert_anthropic_to_multi_messageresponse(
    anthropic_client: AnthropicClient,
    chat_completion: ToolsBetaMessage,
) -> None:
    """Test conversion of Anthropic response to ChatResponse."""
    chat_completion.content.extend(
        [
            TextBlock(
                text="<thinking>I guess I should really get on this</thinking>",
                type="text",
            ),
            ToolUseBlock(
                id="toolu_0000",
                input={"question": "Cats are dead to me?"},
                name="query_gen",
                type="tool_use",
            ),
        ]
    )
    response = anthropic_client.convert_anthropic_to_response(chat_completion)
    assert (
        response
        == ChatResponse(
            id="msg_01Tmqpke9uViGL1uYuoa2H2C",
            cached=False,
            choices=[
                MeadowChoice(
                    index=0,
                    message=ChatMessage(
                        content='<thinking>\nThe user has directly provided a question to ask the query_gen tool:\n"How many cats do I have?"\n\nSince the query_gen tool takes a single required parameter "question" of type string, and the user has provided the question string, we have all the required information to call the tool.\n</thinking>',
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                unparsed_arguments='{"question": "How many cats do I have?"}',
                                name="query_gen",
                            )
                        ],
                    ),
                ),
                MeadowChoice(
                    index=1,
                    message=ChatMessage(
                        content="<thinking>I guess I should really get on this</thinking>",
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                unparsed_arguments='{"question": "Cats are dead to me?"}',
                                name="query_gen",
                            )
                        ],
                    ),
                ),
            ],
            created=response.created,  # hack to match the timestamps as we add time in the conversion
            model="claude-3-opus-20240229",
            usage=MeadowUsage(
                completion_tokens=100, prompt_tokens=495, total_tokens=595
            ),
        )
    )


@pytest.mark.asyncio
async def test_arun_chat(
    anthropic_client: AnthropicClient,
    chat_request: ChatRequest,
    chat_completion: ToolsBetaMessage,
) -> None:
    """Test sending a chat request."""
    # make return value work with await expression
    mock_response = AsyncMock(return_value=chat_completion)
    anthropic_client.client.beta.tools.messages.create = mock_response  # type: ignore
    response = await anthropic_client.arun_chat(chat_request)
    anthropic_client.client.beta.tools.messages.create.assert_called_once_with(
        **anthropic_client.convert_request_for_anthropic(chat_request), system=None
    )
    assert (
        response
        == ChatResponse(
            id="msg_01Tmqpke9uViGL1uYuoa2H2C",
            cached=False,
            choices=[
                MeadowChoice(
                    index=0,
                    message=ChatMessage(
                        content='<thinking>\nThe user has directly provided a question to ask the query_gen tool:\n"How many cats do I have?"\n\nSince the query_gen tool takes a single required parameter "question" of type string, and the user has provided the question string, we have all the required information to call the tool.\n</thinking>',
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                unparsed_arguments='{"question": "How many cats do I have?"}',
                                name="query_gen",
                            )
                        ],
                    ),
                )
            ],
            created=response.created,  # hack to match the timestamps as we add time in the conversion
            model="claude-3-opus-20240229",
            usage=MeadowUsage(
                completion_tokens=100, prompt_tokens=495, total_tokens=595
            ),
        )
    )

    # now test that if the first message is a system message, we pass it as `system`
    chat_request_system = ChatRequest(
        messages=[
            {
                "role": "system",
                "content": "You are a tester",
            },
            {
                "role": "user",
                "content": "Say this is a test",
            },
        ],
        model="gpt-3.5-turbo",
        tools=[
            ToolSpec(
                name="query_gen",
                description="Query generator",
                function_args=[
                    FunctionArgSpec(
                        name="question",
                        description="The question to generate a query for",
                        type="string",
                        required=True,
                    )
                ],
            )
        ],
    )
    mock_response = AsyncMock(return_value=chat_completion)
    anthropic_client.client.beta.tools.messages.create = mock_response  # type: ignore
    response = await anthropic_client.arun_chat(chat_request_system)
    anthropic_client.client.beta.tools.messages.create.assert_called_once_with(
        # We use the original chat_request for conversion here as it doens't have system message
        **anthropic_client.convert_request_for_anthropic(chat_request),
        system="You are a tester",
    )
