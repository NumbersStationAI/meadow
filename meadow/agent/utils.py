import logging
from typing import Callable

from colorama import Fore, Style

from meadow.agent.schema import AgentMessage
from meadow.client.client import Client
from meadow.client.schema import ChatResponse, LLMConfig, ToolSpec

lgger = logging.getLogger(__name__)


class Commands:
    NEXT = "<next>"
    END = "<end>"

    @staticmethod
    def _has_signal_string(content: str, signal_str: str) -> bool:
        """Check if the message contains signalling string."""
        return content.strip().endswith(signal_str) or content.strip().startswith(
            signal_str
        )

    @staticmethod
    def has_next(content: str) -> bool:
        return Commands._has_signal_string(content, Commands.NEXT)

    @staticmethod
    def has_end(content: str) -> bool:
        return Commands._has_signal_string(content, Commands.END)


def print_message(message: AgentMessage, from_agent: str, to_agent: str) -> None:
    """Print a message with color based on the agent."""
    if to_agent == "User":
        content = message.display_content
    else:
        content = message.content
    if from_agent == "User":
        color = Fore.YELLOW
    elif from_agent == "Controller":
        color = Fore.BLUE
    elif from_agent == "Planner":
        color = Fore.RED
    else:
        color = Fore.GREEN
    to_print = f"{from_agent} -> {to_agent}: {content}"
    print(color + to_print + Style.RESET_ALL)


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
    """Generate a reply using autogen.oai."""
    serialized_messages = [system_message.model_dump(include={"role", "content"})]
    serialized_messages += [m.model_dump(include={"role", "content"}) for m in messages]
    # for msg in serialized_messages[1:]:
    #     print(msg["role"], "-----", msg["content"])
    # print("*********")
    chat_response = await client.chat(
        messages=serialized_messages,
        tools=tools,
        model=model,
        seed=llm_config.seed,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        overwrite_cache=overwrite_cache,
    )
    if llm_callback:
        llm_callback(serialized_messages, chat_response)
    # logger.info(
    #     "Ran model",
    #     prompts=serialized_messages,
    #     response=chat_response.choices[0].message,
    #     tools=serialized_tools,
    #     config=self._llm_config,
    # )
    return chat_response
