import logging
from typing import Callable

from termcolor import colored

from meadow.agent.schema import AgentMessage
from meadow.client.client import Client
from meadow.client.schema import ChatResponse, LLMConfig, ToolSpec

lgger = logging.getLogger(__name__)

COLOR_MAP = {
    "User": "yellow",
    "Controller": "light_blue",
    "Planner": "red",
}

AVAILABLE_COLORS = [
    "green",
    "magenta",
    "cyan",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_magenta",
    "light_cyan",
]


def print_message(message: AgentMessage, from_agent: str, to_agent: str) -> None:
    """Print a message with color based on the agent."""
    if to_agent == "User":
        content = message.display_content
    else:
        content = message.content
    if from_agent in COLOR_MAP:
        color = COLOR_MAP[from_agent]
    else:
        # Add a default color
        color = AVAILABLE_COLORS.pop(0)
        COLOR_MAP[from_agent] = color
    to_print = f"{from_agent} -> {to_agent}:\n{content}"
    print(colored(to_print, color))  # type: ignore


async def generate_llm_reply(
    client: Client,
    messages: list[AgentMessage],
    tools: list[ToolSpec],
    system_message: AgentMessage,
    model: str = None,
    llm_config: LLMConfig = LLMConfig(),
    llm_callback: Callable = None,
    overwrite_cache: bool = False,
) -> ChatResponse:
    """Generate a reply using client chat method."""
    # Make sure the chat role is updated wrt to the agent role
    # This should technically be handled in the agents, but if someone
    # forgets to update the role from the agent_role, we do it here
    system_message.role = system_message.agent_role.value
    for message in messages:
        message.role = message.agent_role.value
    # Now dump to dict to pass to client
    serialized_messages = [system_message.model_dump(include={"role", "content"})]
    serialized_messages += [m.model_dump(include={"role", "content"}) for m in messages]
    chat_response = await client.chat(
        messages=serialized_messages,
        tools=tools if tools else None,
        model=model,
        seed=llm_config.seed,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        response_format=llm_config.response_format,
        overwrite_cache=overwrite_cache,
    )
    if llm_callback:
        llm_callback(serialized_messages, chat_response)
    return chat_response
