import logging

from groq import AsyncGroq
from groq.types.chat import ChatCompletion

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


class GroqClient(APIClient):
    """Groq LLM Client."""

    def __init__(self, api_key: str | None) -> None:
        """Initialize the client.

        Args:
            api_key: Groq API key. If no api_key is provided, the client will read
                the GROQ_API_KEY environment variable.
        """
        self.client = AsyncGroq(
            api_key=api_key,
        )

    def convert_request_for_groq(self, request: ChatRequest) -> dict[str, str]:
        """Convert a ChatRequest to a dict for Groq."""
        request_dict = request.model_dump(exclude_none=True)
        keys_to_delete = {"seed"}
        for key in keys_to_delete:
            if key in request_dict:
                del request_dict[key]
        return request_dict

    def convert_groq_to_response(
        self, groq_response: ChatCompletion
    ) -> ChatResponse:
        """Convert an Groq response to a ChatResponse."""
        choices = []
        for i, choice in enumerate(groq_response.choices):
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
            id=groq_response.id,
            cached=False,
            choices=choices,
            created=groq_response.created,
            model=groq_response.model,
            usage=Usage(
                completion_tokens=groq_response.usage.completion_tokens,
                prompt_tokens=groq_response.usage.prompt_tokens,
                total_tokens=groq_response.usage.total_tokens,
            ),
        )
        return response

    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        # convert the request to Groq format
        groq_request = self.convert_request_for_groq(request)
        # send the request to Groq
        groq_response = await self.client.chat.completions.create(
            **groq_request
        )  # type: ignore
        # convert the response to ChatResponse
        response = self.convert_groq_to_response(groq_response)
        return response
