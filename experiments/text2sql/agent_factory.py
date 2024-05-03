import logging

from pydantic import BaseModel

from meadow.agent.agent import Agent
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.user import UserAgent
from meadow.client.client import Client
from meadow.client.schema import ChatMessage, ChatResponse, LLMConfig
from meadow.database.database import Database

logger = logging.getLogger(__name__)


class PromptLog(BaseModel):
    """Prompt log."""

    agent_name: str
    example_idx: int
    messages: list[ChatMessage]
    response: ChatResponse

    def laurel_serialize(self) -> str:
        """Serialize for laurel."""
        final_str = "***\n***\n"
        for msg in self.messages:
            final_str += f"Role: {msg.role}\n"
            final_str += f"{msg.content}\n"
            final_str += "-----\n"
        response_msg = self.response.choices[0].message
        final_str += "~~~~~~\nResponse:\n"
        for k, v in response_msg.model_dump().items():
            final_str += f"Role: {k}\n"
            final_str += f"{v}\n"
            final_str += "-----\n"
        return final_str


def model_callback(
    model_messages: list[dict],
    chat_response: ChatResponse,
    i: int,
    agent_name: str,
    all_prompts: list[list[PromptLog]],
):
    """Call to store messages and response."""
    # Insert the messages into the database
    all_prompts[i].append(
        PromptLog(
            agent_name=agent_name,
            example_idx=i,
            messages=[ChatMessage.model_validate(m) for m in model_messages],
            response=chat_response,
        )
    )


def get_simple_text2sql_agent(
    user_agent: UserAgent,
    client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
    # all_prompts_db: duckdb.DuckDBPyConnection,
    all_prompts_to_save: list,
    example_idx: int,
) -> Agent:
    """Get a simple text2sql agent."""
    callback = lambda model_messages, chat_response: model_callback(
        model_messages,
        chat_response,
        example_idx,
        "SQLGeneratorAgent",
        all_prompts_to_save,
    )
    # prompt_callback = PromptCallback("SQLGeneratorAgent", example_idx, all_prompts_db)
    agent = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
        llm_callback=callback,
    )
    return agent


def get_simple_text2sql_planner_agent(
    user_agent: UserAgent,
    client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
    # all_prompts_db: Path,
    all_prompts_to_save: list,
    example_idx: int,
) -> Agent:
    """Get a simple text2sql agent."""
    callback_sql = lambda model_messages, chat_response: model_callback(
        model_messages,
        chat_response,
        example_idx,
        "SQLGeneratorAgent",
        all_prompts_to_save,
    )
    callback_planner = lambda model_messages, chat_response: model_callback(
        model_messages, chat_response, example_idx, "PlannerAgent", all_prompts_to_save
    )
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=client,
        llm_config=llm_config,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_planner,
    )
    controller = ControllerAgent(user=user_agent, planner=planner, silent=True)
    return controller
