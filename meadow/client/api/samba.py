import json
import logging
import os
import uuid
from typing import Any

import requests

from meadow.client.api.api_client import APIClient
from meadow.client.schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    Usage,
)

logger = logging.getLogger(__name__)


def send_request(api_key: str, url: str, data: dict[str, Any]) -> dict:
    """
    Sends a POST request to the specified URL with headers and JSON data.
    """

    # Headers including the Authorization and Content-Type
    headers = {"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}

    # Sending the POST request
    response = requests.post(url, headers=headers, json=data)

    after_end_pair = response.text.rsplit("event: end_event", 1)
    if len(after_end_pair) != 2:
        raise ValueError("end_event not found in response")
    after_end = after_end_pair[1]
    text = after_end.split("data: ", 1)[1]
    response_dict = json.loads(text)
    return response_dict


class SambaClient(APIClient):
    """Samba LLM Client."""

    def __init__(self, api_key: str | None, url: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: Samba API key. If no api_key is provided, the client will read
                the SAMBA_API_KEY environment variable.
            url: Samba URL. If no url is provided, the client will read the SAMBA_URL
        """
        self.url = url or os.environ.get("SAMBA_URL")
        self.api_key = api_key or os.environ.get("SAMBA_API_KEY")
        if self.url is None:
            raise ValueError(
                "Samba URL not provided. Please pass in or set SAMBA_URL env variable"
            )
        if self.api_key is None:
            raise ValueError(
                "Samba API key not provided. Please pass in or set SAMBA_API_KEY env variable"
            )

    def convert_request_for_samba(self, request: ChatRequest) -> dict[str, str]:
        """Convert a ChatRequest to a dict for Samba."""
        request_dict = request.model_dump(exclude_none=True)
        # perform renamings
        key_renamings = {
            "messages": "inputs",
            "max_tokens": "max_tokens_to_generate",
            "stop": "stop_sequences",
        }
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
            raise NotImplementedError("Tools are not yet supported for Samba.")
        return request_dict

    def convert_samba_to_response(self, samba_response: dict) -> ChatResponse:
        """Convert an Samba response to a ChatResponse."""
        choices = []
        tool_calls = None
        choices.append(
            Choice(
                index=0,
                message=ChatMessage(
                    content=samba_response["completion"],
                    role="assistant",
                    tool_calls=tool_calls,
                ),
            )
        )
        response = ChatResponse(
            # Samba doesn't have an id so assign one
            id=str(uuid.uuid4()),
            cached=False,
            choices=choices,
            created=int(samba_response["start_time"]),
            model=samba_response["model"],
            usage=Usage(
                completion_tokens=samba_response["completion_tokens_count"],
                prompt_tokens=samba_response["prompt_tokens_count"],
                total_tokens=samba_response["total_tokens_count"],
            ),
        )
        return response

    async def arun_chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request."""
        # convert the request to Samba format
        samba_request = self.convert_request_for_samba(request)
        # send the request to Samba
        # TODO: make async
        samba_response = send_request(self.api_key, self.url, samba_request)
        # convert the response to ChatResponse
        response = self.convert_samba_to_response(samba_response)
        return response
