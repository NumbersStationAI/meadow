"""Run agents on text2sql."""

import asyncio
import datetime
import json
import random
import re
from pathlib import Path
from typing import cast

import click
import numpy as np
from pydantic import BaseModel
from rich.console import Console
from tqdm import tqdm

from experiments.text2sql.agent_factory import (
    PromptLog,
    get_simple_text2sql_agent,
    get_simple_text2sql_planner_agent,
)
from experiments.text2sql.eval_user import EvalUserAgent
from experiments.text2sql.utils import load_data, read_tables_json
from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage
from meadow.cache.duckdb import DuckDBCache
from meadow.client.api.anthropic import AnthropicClient
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.connector.connector import Table
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database
from meadow.history.message_history import MessageHistory

console = Console(soft_wrap=True)


class TextToSQLParams(BaseModel):
    """Text to SQL parameters."""

    instruction: str
    agent_type: str
    db_id: str
    database_path: str


async def agenerate_sql(
    text_to_sql_in: list[TextToSQLParams],
    client: Client,
    llm_config: LLMConfig,
    overwrite_cache: bool,
    # all_prompts_db: duckdb.DuckDBPyConnection,
) -> tuple[list[Agent], list[list[PromptLog]]]:
    """Async SQL generation."""
    text_to_sql_for_gather = []
    agents_gather = []
    all_prompts_to_save: list[list[PromptLog]] = [[] for _ in text_to_sql_in]
    for i, text_to_sql in enumerate(text_to_sql_in):
        # load database
        connector = SQLiteConnector(text_to_sql.database_path)
        database = Database(connector=connector)

        # load "user" agent
        user_agent = EvalUserAgent(
            name="User",
            silent=True,
            description="User agent",
        )

        # load agent
        if text_to_sql.agent_type == "text2sql":
            agent = get_simple_text2sql_agent(
                user_agent=user_agent,
                client=client,
                llm_config=llm_config,
                database=database,
                overwrite_cache=overwrite_cache,
                all_prompts_to_save=all_prompts_to_save,
                example_idx=i,
            )
        elif text_to_sql.agent_type == "text2sql_with_plan":
            agent = get_simple_text2sql_planner_agent(
                user_agent=user_agent,
                client=client,
                llm_config=llm_config,
                database=database,
                overwrite_cache=overwrite_cache,
                all_prompts_to_save=all_prompts_to_save,
                example_idx=i,
            )
        else:
            raise ValueError(f"Unknown agent type {text_to_sql.agent_type}")
        agents_gather.append(agent)
        instruction = text_to_sql.instruction

        message = AgentMessage(
            role="user", content=instruction, generating_agent=user_agent.name
        )
        text_to_sql_for_gather.append(agent.receive(message, user_agent))
    await asyncio.gather(*text_to_sql_for_gather)
    return agents_gather, all_prompts_to_save


def generate_sql(
    text_to_sql_in: list[TextToSQLParams],
    client: Client,
    llm_config: LLMConfig,
    overwrite_cache: bool,
    async_batch_size: int,
) -> tuple[list[AgentMessage], list[list[PromptLog]]]:
    """Ask agent to generate SQL."""
    # Batch inputs for asyncio
    text_to_sql_in_batches = [
        text_to_sql_in[i : i + async_batch_size]
        for i in range(0, len(text_to_sql_in), async_batch_size)
    ]
    agents = []
    all_prompts_list = []
    for batch in tqdm(text_to_sql_in_batches):
        response_agent_batch, all_prompts_batch = asyncio.run(
            agenerate_sql(
                batch,
                client=client,
                llm_config=llm_config,
                overwrite_cache=overwrite_cache,
            )
        )
        agents.extend(response_agent_batch)
        all_prompts_list.extend(all_prompts_batch)

    assert len(agents) == len(all_prompts_list)
    # Extract the sql responses from each agent chat
    sql_responses = []
    for i in range(len(agents)):
        agent = agents[i]
        # Hack to get text2sql agent chat
        try:
            t2s = list(agent._planner._available_agents.values())[0]
            messages = cast(MessageHistory, agent._messages).get_messages(t2s)
        except AttributeError:
            evuse = list(agent._messages.get_all_messages().keys())[0]
            messages = list(cast(MessageHistory, agent._messages).get_messages(evuse))
        last_sql_message = None
        for msg in messages[::-1]:
            if "SQL:" in msg.display_content:
                last_sql_message = msg
                break
        if last_sql_message is None:
            sql = ""
        else:
            sql = (
                last_sql_message.display_content.split("Table:")[0]
                .strip()[len("SQL:") :]
                .strip()
            )
        sql_responses.append(sql)

    assert len(sql_responses) == len(text_to_sql_in)
    return sql_responses, all_prompts_list


def get_text_to_sql_in(
    input_question: dict,
    agent_type: str,
    db_to_tables: dict[str, dict[str, Table]],
    database_path: Path,
) -> TextToSQLParams:
    """Format input question for text2sql function."""
    question = input_question["question"]
    db_id = input_question.get("db_id", None)
    assert db_id in db_to_tables, f"db_id {db_id} not in db_to_tables"

    text_to_sql_in = TextToSQLParams(
        instruction=question,
        agent_type=agent_type,
        db_id=db_id,
        database_path=str(database_path / db_id / f"{db_id}.sqlite"),
    )
    return text_to_sql_in


@click.group()
def cli() -> None:
    """Entrypoint."""
    pass


@cli.command()
@click.argument("dataset-path")
@click.argument("table-meta-path")
@click.argument("database-path")
# Run options
@click.option("--output-dir", type=str, default="")
@click.option("--run-name", type=str, default="")
@click.option("--num-run", type=int, default=-1)
@click.option("--num-print", type=int, default=20)
@click.option("--async-batch-size", type=int, default=20)
# Data options
@click.option("--lowercase-schema", is_flag=True, default=False)
# Agent options
@click.option("--agent-type", type=str, default="text2sql")
# Model options
@click.option("--max-tokens", type=int, default=1000)
@click.option("--temperature", type=float, default=0)
# Client options
@click.option("--model", type=str, default="claude-3-opus-20240229")
@click.option(
    "--client-cache-path",
    type=str,
    default="/home/lorr1/projects/code/meadow/test_cache.duckdb",
)
@click.option("--api-key", type=str, required=True)
@click.option("--overwrite-cache", is_flag=True, default=False)
def predict(
    dataset_path: str,
    table_meta_path: str,
    database_path: str,
    output_dir: str,
    run_name: str,
    num_run: int,
    num_print: int,
    async_batch_size: int,
    lowercase_schema: bool,
    agent_type: str,
    max_tokens: int,
    temperature: float,
    model: str,
    client_cache_path: str,
    api_key: str,
    overwrite_cache: bool,
) -> None:
    """Predict SQL."""
    random.seed(0)
    np.random.seed(0)
    locals_dict = locals()
    console.print(json.dumps(locals_dict, indent=2))

    # load client
    llm_config = LLMConfig(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cache = DuckDBCache(client_cache_path)
    client = Client(
        cache=cache,
        api_client=AnthropicClient(api_key=api_key),
        model=model,
    )

    console.print("Loading metadata...")
    db_to_tables = read_tables_json(table_meta_path, lowercase=lowercase_schema)

    console.print("Loading data...")
    data = load_data(dataset_path)
    if num_run > 0:
        console.print(f"Running on {min(len(data), num_run)} examples")
        data = data[:num_run]
    original_data = data

    # load the examples
    console.print("Formatting data...")
    num_print = min(num_print, len(data))
    text_to_sql_in = [
        get_text_to_sql_in(
            input_question=input_question,
            agent_type=agent_type,
            db_to_tables=db_to_tables,
            database_path=Path(database_path),
        )
        for input_question in data
    ]

    # Get output filename
    full_dataset_path = Path(dataset_path)
    # Get todays date
    date_today = datetime.datetime.now().strftime("%y-%m-%d")
    if run_name:
        run_name = f"{run_name}_"
    suffix = f"{run_name}{full_dataset_path.stem}_{date_today}.json"  # noqa: E501
    prefix = f"{agent_type}_{model}"
    output_filename = f"{prefix}_{suffix}"
    console.print(f"Saving to {Path(output_dir) / output_filename}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / output_filename, "w") as fout:
        for i in range(num_print):
            # Run a few to get some examples to console.print
            predictions, prompts = generate_sql(
                text_to_sql_in=[text_to_sql_in[i]],
                client=client,
                llm_config=llm_config,
                overwrite_cache=overwrite_cache,
                async_batch_size=async_batch_size,
            )
            for prediction, prompt in zip(predictions, prompts):
                prediction = re.sub(r"[\s\t\n]+", " ", prediction)
                console.print(f"[blue]Prompt:[/blue] {json.dumps(prompt, indent=2)}")
                console.print(f"[red]Prediction:[/red] {prediction}")
                if data[i].get("query") or data[i].get("sql"):
                    console.print(
                        "[purple]Gold:[/purple] "
                        f"{data[i].get('query') or data[i].get('sql')}"
                    )
                console.print("\n****\n")

        # Run the entire thing now - the to_console.print results will be in cache and fast
        generated_sqls, generated_prompts = generate_sql(
            text_to_sql_in=text_to_sql_in,
            client=client,
            llm_config=llm_config,
            overwrite_cache=overwrite_cache,
            async_batch_size=async_batch_size,
        )

        for i in range(len(generated_sqls)):
            prediction = generated_sqls[i]
            prompt = [m.laurel_serialize() if m else None for m in generated_prompts[i]]
            entry = {
                **original_data[i],
                "pred": prediction,
                "prompt": prompt,
            }
            print(json.dumps(entry), file=fout)
    console.print(f"Saved to {Path(output_dir) / output_filename}")


if __name__ == "__main__":
    cli()
