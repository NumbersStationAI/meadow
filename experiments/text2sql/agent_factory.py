import logging
from typing import Callable

from pydantic import BaseModel

from meadow.agent.agent import Agent
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.attribute_detector import AttributeDetectorAgent
from meadow.agent.data_agents.planner_constraints import attribute_detector_constraints, sql_agent_constraints
from meadow.agent.data_agents.schema_renamer import SchemaRenamerAgent
from meadow.agent.data_agents.sql_decomposer import SQLDecomposerAgent
from meadow.agent.data_agents.sql_planner import SQLPlannerAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.executor.data_executors.empty_result_debugger import (
    EmptyResultExecutor,
)
from meadow.agent.executor.data_executors.sql_validate_reask import SQLValidateExecutor
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


def get_text2sql_agent(
    benchmark: str,
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
    all_prompts_to_save: list,
    example_idx: int,
    add_reask: bool,
    add_empty_table: bool,
    add_decomposer: bool,
    add_sql_planner: bool,
    add_attribute_selector: bool,
    add_schema_cleaner: bool,
) -> ControllerAgent:
    """Get a text2sql agent with optional components."""
    callback_sql: Callable = lambda model_messages, chat_response: model_callback(
        model_messages, chat_response, example_idx, "SQLGeneratorAgent", all_prompts_to_save
    )

    executors = [
        SQLValidateExecutor(
            client=None,
            llm_config=None,
            database=database,
            max_execution_attempts=0 if not add_reask else 2,
        ),
    ]

    if add_empty_table:
        callback_empty: Callable = lambda model_messages, chat_response: model_callback(
            model_messages, chat_response, example_idx, "EmptyResultExecutor", all_prompts_to_save
        )
        executors.append(
            EmptyResultExecutor(
                client=client,
                llm_config=llm_config,
                database=database,
                llm_callback=callback_empty,
            )
        )

    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=executors,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )

    agents = [text2sql]

    if add_decomposer:
        callback_decomp: Callable = lambda model_messages, chat_response: model_callback(
            model_messages, chat_response, example_idx, "DecomposerAgent", all_prompts_to_save
        )
        text2sql_decomposer = SQLDecomposerAgent(
            client=planner_client,
            llm_config=llm_config,
            database=database,
            available_agents=[text2sql],
            llm_callback=callback_decomp,
        )
        agents.insert(0, text2sql_decomposer)

    if add_sql_planner:
        callback_planner: Callable = lambda model_messages, chat_response: model_callback(
            model_messages, chat_response, example_idx, "PlannerAgent", all_prompts_to_save
        )
        sql_planner = SQLPlannerAgent(
            client=planner_client,
            llm_config=llm_config,
            database=database,
            overwrite_cache=overwrite_cache,
            llm_callback=callback_planner,
        )
        agents.insert(0, sql_planner)

    if add_attribute_selector:
        callback_attr: Callable = lambda model_messages, chat_response: model_callback(
            model_messages, chat_response, example_idx, "AttributeDetectorAgent", all_prompts_to_save
        )
        attribute_detector = AttributeDetectorAgent(
            client=client,
            llm_config=llm_config,
            database=database,
            llm_callback=callback_attr,
        )
        agents.insert(0, attribute_detector)

    if add_schema_cleaner:
        callback_clean: Callable = lambda model_messages, chat_response: model_callback(
            model_messages, chat_response, example_idx, "SchemaCleanerAgent", all_prompts_to_save
        )
        schema_cleaner = SchemaRenamerAgent(
            client=client,
            llm_config=llm_config,
            database=database,
            llm_callback=callback_clean,
        )
        agents.insert(0, schema_cleaner)

    callback_planner: Callable = lambda model_messages, chat_response: model_callback(
        model_messages, chat_response, example_idx, "PlannerAgent", all_prompts_to_save
    )
    planner = PlannerAgent(
        available_agents=agents,
        client=None if len(agents) == 1 else client,
        llm_config=llm_config,
        database=database,
        constraints=[attribute_detector_constraints, sql_agent_constraints],
        overwrite_cache=overwrite_cache,
        llm_callback=callback_planner,
    )

    controller = ControllerAgent(
        supervisor=user_agent, planner=planner, database=database, silent=True
    )

    return controller
