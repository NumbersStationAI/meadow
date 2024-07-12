import logging

from together import AsyncTogether
from together.types.chat_completions import ChatCompletionResponse

from meadow.client.api.api_client import APIClient
from meadow.client.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


class TogetherClient(APIClient):
    """Together LLM Client."""

    def __init__(self, api_key: str | None) -> None:
        """Initialize the client.

        Args:
            api_key: Together API key. If no api_key is provided, the client will read
                the TOGETHER_API_KEY environment variable.
        """
        self.client = AsyncTogether(
            api_key=api_key,
        )

    def convert_request_for_together(self, request: ChatRequest) -> dict[str, str]:
        """Convert a ChatRequest to a dict for Together."""
        request_dict = request.model_dump(exclude_none=True)
        keys_to_delete = {"seed"}
        for key in keys_to_delete:
            if key in request_dict:
                del request_dict[key]
        return request_dict

    def convert_together_to_response(
        self, together_response: ChatCompletionResponse
    ) -> ChatResponse:
        """Convert an Together response to a ChatResponse."""
        choices = []
        for i, choice in enumerate(together_response.choices):
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tool_call.function.name,
                        unparsed_arguments=tool_call.function.arguments,
                    )
                    for tool_call in choice.message.tool_calls
                ]
            choices.append(
                Choice(
                    index=i,
                    message=ChatMessage(
                        content=choice.message.content,
                        role=choice.message.role.value,
                        tool_calls=tool_calls,
                    ),
                )
            )
        response = ChatResponse(
            id=together_response.id,
            cached=False,
            choices=choices,
            created=together_response.created,
            model=together_response.model,
            usage=Usage(
                completion_tokens=together_response.usage.completion_tokens,
                prompt_tokens=together_response.usage.prompt_tokens,
                total_tokens=together_response.usage.total_tokens,
            ),
        )
        return response

    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        # convert the request to Together format
        together_request = self.convert_request_for_together(request)
        # send the request to Together
        together_response = await self.client.chat.completions.create(
            **together_request
        )  # type: ignore
        # convert the response to ChatResponse
        response = self.convert_together_to_response(together_response)
        return response
