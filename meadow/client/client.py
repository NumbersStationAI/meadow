"""Model client with cache calls."""

from meadow.cache.cache import Cache
from meadow.client.api.api_client import APIClient
from meadow.client.schema import ChatRequest, ChatResponse, ToolSpec


class Client:
    """Model client class."""

    def __init__(
        self,
        api_client: APIClient,
        model: str | None = None,
        cache: Cache | None = None,
    ) -> None:
        """Initialize the client.

        If set, model will be the default model used for chat requests.
        """
        if not api_client:
            raise ValueError("API client is required.")
        self.cache = cache
        self.api_client = api_client
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = None,
        n: int = 1,
        seed: int = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        top_k: int = None,
        stop: list[str] = None,
        presence_penalty: float = None,
        frequency_penalty: float = None,
        # { "type": "json_object" } for JSON output
        response_format: dict[str, str] = None,
        tools: list[ToolSpec] = None,
        overwrite_cache: bool = False,
    ) -> ChatResponse:
        """Run chat request against client.

        See schema.ChatRequest for parameter descriptions.

        If cache is not None and overwrite_cache is False, the cache will be
        checked first and returned if available.
        """
        if not messages:
            raise ValueError("Messages are required.")
        if not model and not self.model:
            raise ValueError("Model is required.")
        model = model or self.model
        # build the ChatRequest
        request = ChatRequest(
            messages=messages,
            model=model,
            n=n,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            response_format=response_format,
            tools=tools,
        )
        # check cache
        if self.cache and not overwrite_cache:
            cached_response = self.cache.get(request)
            if cached_response:
                return cached_response
        # run chat
        response = await self.api_client.arun_chat(request)
        # cache response
        if self.cache:
            self.cache.set(request, response)
        return response
