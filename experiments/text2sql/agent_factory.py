import json
import logging
from pathlib import Path

import duckdb
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


class PromptCallback:
    """Prompt callback."""

    def __init__(
        self,
        agent_name: str,
        example_idx: int,
        all_prompts_db: duckdb.DuckDBPyConnection,
    ):
        """Initialize."""
        self.agent_name = agent_name
        self.example_idx = example_idx
        self.all_prompts_db = all_prompts_db
        self._create_table(agent_name)

    def _create_table(self, table_name: str):
        # Create table for json data of not exist
        self.all_prompts_db.begin()
        self.all_prompts_db.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (example_idx INTEGER, messages JSON, response JSON)"
        )
        self.all_prompts_db.commit()

    def __call__(self, model_messages: list[dict], chat_response: ChatResponse):
        """Call to store messages and response."""
        # Insert the messages into the database
        self.all_prompts_db.begin()
        self.all_prompts_db.execute(
            f"INSERT INTO {self.agent_name} VALUES (?, ?, ?)",
            (
                self.example_idx,
                model_messages,
                chat_response.model_dump(),
            ),
        )
        self.all_prompts_db.commit()

    def drop_data(self):
        """Close the database."""
        self.all_prompts_db.begin()
        self.all_prompts_db.execute(f"DROP TABLE {self.agent_name}")
        self.all_prompts_db.commit()

    def get_all_prompts(self) -> list[PromptLog]:
        """Get all the prompts."""
        results = self.all_prompts_db.execute(
            f"SELECT * FROM {self.agent_name}"
        ).fetchall()
        parsed_results: list[PromptLog] = []
        for result in results:
            ex_idx, messages, response = result
            parsed_results.append(
                PromptLog(
                    agent_name=self.agent_name,
                    example_idx=ex_idx,
                    messages=[
                        ChatMessage.model_validate(m) for m in json.loads(messages)
                    ],
                    response=ChatResponse.model_validate_json(response),
                )
            )
        return sorted(parsed_results, key=lambda x: x.example_idx)


def get_simple_text2sql_agent(
    user_agent: UserAgent,
    client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
    all_prompts_db: duckdb.DuckDBPyConnection,
    example_idx: int,
) -> tuple[Agent, list[PromptCallback]]:
    """Get a simple text2sql agent."""
    prompt_callback = PromptCallback("SQLGeneratorAgent", example_idx, all_prompts_db)
    agent = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
        llm_callback=prompt_callback,
    )
    return agent, [prompt_callback]


def get_simple_text2sql_planner_agent(
    user_agent: UserAgent,
    client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
    all_prompts_db: Path,
    example_idx: int,
) -> tuple[Agent, list[PromptCallback]]:
    """Get a simple text2sql agent."""
    sql_prompt_callback = PromptCallback(
        "SQLGeneratorAgent", example_idx, all_prompts_db
    )
    planner_prompt_callback = PromptCallback(
        "PlannerAgent", example_idx, all_prompts_db
    )
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
        llm_callback=sql_prompt_callback,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=client,
        llm_config=llm_config,
        overwrite_cache=overwrite_cache,
        llm_callback=planner_prompt_callback,
    )
    controller = ControllerAgent(user=user_agent, planner=planner, silent=True)
    return controller, [planner_prompt_callback, sql_prompt_callback]
