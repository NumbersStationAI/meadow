import json
import time

import structlog
from anthropic import AsyncAnthropic
from anthropic.types.beta.tools.tools_beta_message import ToolsBetaMessage

from meadow.agent.schema import ChatMessage, ToolCall
from meadow.client.api.api_client import APIClient
from meadow.client.schema import ChatRequest, ChatResponse, Choice, Usage

logger = structlog.getLogger(__name__)

DEFAULT_MAX_TOKENS = 100


class AnthropicClient(APIClient):
    """Anthropic LLM Client."""

    def __init__(self, api_key: str | None) -> None:
        """Initialize the client.

        Args:
            api_key: Anthropic API key. If no api_key is provided, the client will read
                the ANTHROPIC_API_KEY environment variable.
        """
        self.client = AsyncAnthropic(
            api_key=api_key,
        )

    def convert_request_for_anthropic(self, request: ChatRequest) -> dict[str, str]:
        """Convert a ChatRequest to a dict for Anthropic.

        Anthropic requires max_tokens to be set. If None is provided, will default.
        """
        request_dict = request.model_dump(exclude_none=True)
        # perform renamings
        key_renamings = {"stop": "stop_sequences"}
        keys_to_delete = {
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
        }
        for old_key, new_key in key_renamings.items():
            if old_key in request_dict:
                request_dict[new_key] = request_dict.pop(old_key)
        for key in keys_to_delete:
            if key in request_dict:
                del request_dict[key]
        # now reformat tools to fit Anthropic's format
        if "tools" in request_dict:
            for i, tool in enumerate(request_dict["tools"]):
                tool = tool["function"]
                tool["input_schema"] = tool.pop("parameters")
                request_dict["tools"][i] = tool
        if "max_tokens" not in request_dict:
            request_dict["max_tokens"] = DEFAULT_MAX_TOKENS
        return request_dict

    def convert_anthropic_to_response(
        self, anthropic_response: ToolsBetaMessage
    ) -> ChatResponse:
        """Convert an Anthropic response to a ChatResponse."""
        choices = []
        tool_calls = []
        current_choice = None
        choice_index = 0
        # N.B. unclear yet why content would have multiple 'text' responses, but
        # parsing them anyways
        for choice in anthropic_response.content:
            if choice.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=choice.name,
                        arguments=json.dumps(choice.input),
                    )
                )
            elif choice.type == "text":
                if current_choice is not None:
                    choices.append(
                        Choice(
                            index=choice_index,
                            message=ChatMessage(
                                content=current_choice,
                                role=anthropic_response.role,
                                tool_calls=tool_calls if tool_calls else None,
                            ),
                        )
                    )
                    tool_calls = []
                    choice_index += 1
                current_choice = choice.text
            else:
                raise ValueError(f"Unknown choice type: {choice.type}")
        if current_choice is not None:
            choices.append(
                Choice(
                    index=choice_index,
                    message=ChatMessage(
                        content=current_choice,
                        role=anthropic_response.role,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                )
            )
        response = ChatResponse(
            id=anthropic_response.id,
            cached=False,
            choices=choices,
            created=int(time.time()),
            model=anthropic_response.model,
            usage=Usage(
                completion_tokens=anthropic_response.usage.output_tokens,
                prompt_tokens=anthropic_response.usage.input_tokens,
                total_tokens=anthropic_response.usage.output_tokens
                + anthropic_response.usage.input_tokens,
            ),
        )
        return response

    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        # convert the request to Anthropic format
        anthropic_request = self.convert_request_for_anthropic(request)
        # send the request to Anthropic
        anthropic_response = await self.client.beta.tools.messages.create(
            **anthropic_request
        )
        # convert the response to ChatResponse
        response = self.convert_anthropic_to_response(anthropic_response)
        return response
