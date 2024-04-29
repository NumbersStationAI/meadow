from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from meadow.client.api.api_client import APIClient
from meadow.client.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ToolCall,
    Usage,
)


class OpenAIClient(APIClient):
    """OpenAI LLM Client."""

    def __init__(self, api_key: str | None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenAI API key. If no api_key is provided, the client will read
                the OPENAI_API_KEY environment variable.
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
        )

    def convert_request_for_openai(self, request: ChatRequest) -> dict[str, str]:
        """Convert a ChatRequest to a dict for OpenAI."""
        request_dict = request.model_dump(exclude_none=True)
        return request_dict

    def convert_openai_to_response(
        self, openai_response: ChatCompletion
    ) -> ChatResponse:
        """Convert an OpenAI response to a ChatResponse."""
        choices = []
        for i, choice in enumerate(openai_response.choices):
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
                        role=choice.message.role,
                        tool_calls=tool_calls,
                    ),
                )
            )
        response = ChatResponse(
            id=openai_response.id,
            cached=False,
            choices=choices,
            created=openai_response.created,
            model=openai_response.model,
            usage=Usage(
                completion_tokens=openai_response.usage.completion_tokens,
                prompt_tokens=openai_response.usage.prompt_tokens,
                total_tokens=openai_response.usage.total_tokens,
            ),
        )
        return response

    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        # convert the request to OpenAI format
        openai_request = self.convert_request_for_openai(request)
        # send the request to OpenAI
        openai_response = await self.client.chat.completions.create(**openai_request)  # type: ignore
        # convert the response to ChatResponse
        response = self.convert_openai_to_response(openai_response)
        return response
