import argparse
import asyncio

from meadow import Client
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.user import UserAgent
from meadow.cache import DuckDBCache
from meadow.client.api.anthropic import AnthropicClient
from meadow.client.schema import LLMConfig
from meadow.database.connector.duckdb import DuckDBConnector
from meadow.database.database import Database


def run_meadow(api_key: str, duckdb_path: str, instruction: str) -> None:
    """Main."""
    if not instruction:
        instruction = input("Enter a question: ")

    # Build database
    connector = DuckDBConnector(duckdb_path)
    database = Database(connector)

    # Build LLM client
    cache = DuckDBCache("test_cache.duckdb")
    client = Client(
        cache=cache,
        api_client=AnthropicClient(api_key=api_key),
        model="claude-3-opus-20240229",
    )

    # Build Agents
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=LLMConfig(
            max_tokens=500,
            temperature=0.0,
        ),
        database=database,
        silent=False,
    )
    user = UserAgent(name="User", silent=False)
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=client,
        llm_config=LLMConfig(
            max_tokens=500,
            temperature=0.0,
        ),
        database=database,
        silent=False,
    )
    controller = ControllerAgent(
        user=user,
        planner=planner,
        silent=False,
    )

    # Start the task
    asyncio.run(controller.initiate_chat(instruction))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run the Meadow demo.")
    argparser.add_argument(
        "--api-key",
        type=str,
        help="The API key to use for the Anthropic API.",
        required=True,
    )
    argparser.add_argument(
        "--duckdb-path",
        type=str,
        help="The path to the DuckDB database.",
        default="examples/sales_example.duckdb",
    )
    argparser.add_argument(
        "--instruction",
        type=str,
        help="The instruction to send to the controller agent.",
        default=None,
    )
    args = argparser.parse_args()
    run_meadow(args.api_key, args.duckdb_path, args.instruction)
