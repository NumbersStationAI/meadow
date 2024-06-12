import argparse
import asyncio

from meadow import Client
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.attribute_detector import AttributeDetectorAgent
from meadow.agent.data_agents.planner_constraints import (
    attribute_detector_constraints,
    sql_agent_constraints,
)
from meadow.agent.data_agents.schema_renamer import SchemaRenamerAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.user import UserAgent
from meadow.agent.utils import print_message
from meadow.cache import DuckDBCache
from meadow.client.api.anthropic import AnthropicClient
from meadow.client.api.api_client import APIClient
from meadow.client.api.openai import OpenAIClient
from meadow.client.schema import LLMConfig
from meadow.database.connector.connector import Connector
from meadow.database.connector.duckdb import DuckDBConnector
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database


def get_full_text2sql_agent(
    user_agent: UserAgent,
    client: Client,
    big_client: Client,
    llm_config: LLMConfig,
    database: Database,
    auto_advance: bool,
    overwrite_cache: bool,
) -> ControllerAgent:
    """Get a full text2sql agent."""
    attribute_detector = AttributeDetectorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    schema_cleaner = SchemaRenamerAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    planner = PlannerAgent(
        available_agents=[schema_cleaner, attribute_detector, text2sql],
        client=big_client,
        llm_config=llm_config,
        database=database,
        constraints=[attribute_detector_constraints, sql_agent_constraints],
        overwrite_cache=overwrite_cache,
    )
    controller = ControllerAgent(
        supervisor=user_agent,
        planner=planner,
        database=database,
        silent=True,
        supervisor_auto_respond=auto_advance,
    )
    return controller


def run_meadow(
    api_provider: str,
    api_key: str,
    db_type: str,
    db_path: str,
    model: str,
    instruction: str,
    auto_advance: bool,
) -> None:
    """Main."""
    if not instruction:
        instruction = input("Enter a question: ")

    # Build database
    if db_type == "duckdb":
        connector: Connector = DuckDBConnector(db_path)
    elif db_type == "sqlite":
        connector = SQLiteConnector(db_path)
    else:
        raise ValueError(f"Unknown database type: {db_type}")
    database = Database(connector)

    # Build LLM client
    cache = DuckDBCache("test_cache.duckdb")

    if api_provider == "anthropic":
        api_client: APIClient = AnthropicClient(api_key=api_key)
    elif api_provider == "openai":
        api_client = OpenAIClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")
    client = Client(
        cache=cache,
        api_client=api_client,
        model=model,
    )
    big_client = Client(cache=cache, api_client=api_client, model="gpt-4o")
    llm_config = LLMConfig(
        max_tokens=1000,
        temperature=0.0,
    )
    user = UserAgent(name="User", silent=False)
    controller = get_full_text2sql_agent(
        user_agent=user,
        client=client,
        big_client=big_client,
        llm_config=llm_config,
        database=database,
        auto_advance=auto_advance,
        overwrite_cache=False,
    )
    asyncio.run(controller.initiate_chat(instruction))
    all_messages = controller.get_messages(user)
    print("****FINAL MESSAGES TO/FROM USER****")
    for msg in all_messages:
        print_message(msg, msg.sending_agent, msg.receiving_agent)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Meadow demo.")
    argparser.add_argument(
        "--api-provider",
        type=str,
        help="The API provider to use.",
        default="openai",
    )
    argparser.add_argument(
        "--api-key",
        type=str,
        help="The API key to use.",
        default=None,
    )
    argparser.add_argument(
        "--db-type",
        type=str,
        help="The type od DB: duckdb or sqlite.",
        default="sqlite",
    )
    argparser.add_argument(
        "--db-path",
        type=str,
        help="The path to the database.",
        default="examples/data/database_sqlite/sales_ambiguous_joins_example/sales_ambiguous_joins_example.sqlite",
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="Anthropic model.",
        default="gpt-4o",
    )
    argparser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Automatically advance the conversation.",
    )
    argparser.add_argument(
        "--instruction",
        type=str,
        help="The instruction to send to the controller agent.",
        default=None,
    )
    args = argparser.parse_args()
    run_meadow(
        args.api_provider,
        args.api_key,
        args.db_type,
        args.db_path,
        args.model,
        args.instruction,
        args.auto_advance,
    )
