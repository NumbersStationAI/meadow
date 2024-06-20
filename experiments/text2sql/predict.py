"""Run agents on text2sql."""

import asyncio
import copy
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

from experiments.text2sql.agent_factory import PromptLog, get_text2sql_agent
from experiments.text2sql.eval_user import EvalUserAgent
from experiments.text2sql.utils import load_data, read_tables_json
from meadow.agent.agent import Agent
from meadow.cache.duckdb import DuckDBCache
from meadow.client.api.anthropic import AnthropicClient
from meadow.client.api.openai import OpenAIClient
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.connector.connector import Table
from meadow.database.connector.duckdb import DuckDBConnector
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database
from meadow.history.message_history import MessageHistory

console = Console(soft_wrap=True)


class TextToSQLParams(BaseModel):
    """Text to SQL parameters."""

    benchmark: str
    instruction: str
    db_id: str
    database_path: str
    add_reask: bool = False
    add_empty_table: bool = False
    add_decomposer: bool = False
    add_sql_planner: bool = False
    add_attribute_selector: bool = False
    add_schema_cleaner: bool = False


async def agenerate_sql(
    text_to_sql_in: list[TextToSQLParams],
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    real_tables: dict[str, dict[str, Table]] | None,
    overwrite_cache: bool,
    # all_prompts_db: duckdb.DuckDBPyConnection,
) -> tuple[list[Agent], list[list[PromptLog]]]:
    """Async SQL generation."""
    text_to_sql_for_gather = []
    agents_gather = []
    all_prompts_to_save: list[list[PromptLog]] = []
    for i, text_to_sql in enumerate(text_to_sql_in):
        all_prompts_to_save.append([])
        # load database
        if Path(text_to_sql.database_path).suffix == ".sqlite":
            connector = SQLiteConnector(text_to_sql.database_path)
        else:
            connector = DuckDBConnector(text_to_sql.database_path)
        database = Database(connector=connector)
        # hack to use the given schema information in benchmarks if available
        if real_tables:
            database._base_tables = copy.deepcopy(real_tables[text_to_sql.db_id])
        # load "user" agent
        user_agent = EvalUserAgent(
            name="User",
            silent=True,
            description="User agent",
        )

        # load agent
        agent = get_text2sql_agent(
            benchmark=text_to_sql.benchmark,
            user_agent=user_agent,
            client=client,
            planner_client=planner_client,
            llm_config=llm_config,
            database=database,
            overwrite_cache=overwrite_cache,
            all_prompts_to_save=all_prompts_to_save,
            example_idx=i,
            add_reask=text_to_sql.add_reask,
            add_empty_table=text_to_sql.add_empty_table,
            add_decomposer=text_to_sql.add_decomposer,
            add_sql_planner=text_to_sql.add_sql_planner,
            add_attribute_selector=text_to_sql.add_attribute_selector,
            add_schema_cleaner=text_to_sql.add_schema_cleaner,
        )
        agents_gather.append(agent)
        instruction = text_to_sql.instruction
        text_to_sql_for_gather.append(agent.initiate_chat(instruction))
    await asyncio.gather(*text_to_sql_for_gather)
    return agents_gather, all_prompts_to_save


def generate_sql(
    text_to_sql_in: list[TextToSQLParams],
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    real_tables: dict[str, dict[str, Table]],
    overwrite_cache: bool,
    async_batch_size: int,
) -> tuple[list[tuple[str, str]], list[list[PromptLog]]]:
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
                planner_client=planner_client,
                llm_config=llm_config,
                real_tables=real_tables,
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
        database: Database = agent._planner._database
        # Hack to get text2sql and user agent chat
        evuse = None
        for ag in agent._messages.get_all_messages().keys():
            if "user" in ag.name.lower():
                evuse = ag
        assert evuse
        messages = list(cast(MessageHistory, agent._messages).get_messages(evuse))
        last_sql_message = None

        for msg in messages[::-1]:
            if "SQL:" in msg.display_content:
                last_sql_message = msg
                break
        if last_sql_message is None:
            print("DID NOT FIND SQL FOR")
            for msg in messages:
                print(msg.content[:200])
            sql = ""
            tbl = ""
        else:
            sql = (
                last_sql_message.display_content.split("Table:")[0]
                .strip()[len("SQL:") :]
                .strip()
            )
            # noramlize sql with the views
            sql = database.normalize_query(sql)
            if "Table:" in last_sql_message.display_content:
                tbl = last_sql_message.display_content.split("Table:")[1].split("Warning:")[0].strip()
            else:
                tbl = ""
        database.close()
        # print("FINAL SQL", sql, "\n\n", tbl)
        sql_responses.append((sql, tbl))

    assert len(sql_responses) == len(text_to_sql_in)
    return sql_responses, all_prompts_list


def get_text_to_sql_in(
    benchmark: str,
    input_question: dict,
    db_to_tables: dict[str, dict[str, Table]],
    database_path: Path,
    add_reask: bool,
    add_empty_table: bool,
    add_decomposer: bool,
    add_sql_planner: bool,
    add_attribute_selector: bool,
    add_schema_cleaner: bool,
) -> TextToSQLParams:
    """Format input question for text2sql function."""
    question = input_question["question"]
    db_id = input_question.get("db_id", None)
    assert db_id in db_to_tables, f"db_id {db_id} not in db_to_tables"

    possible_db_path = database_path / db_id / f"{db_id}.sqlite"
    if not possible_db_path.exists():
        possible_db_path = database_path / db_id / f"{db_id}.duckdb"

    text_to_sql_in = TextToSQLParams(
        benchmark=benchmark,
        instruction=question,
        db_id=db_id,
        database_path=str(possible_db_path),
        add_reask=add_reask,
        add_empty_table=add_empty_table,
        add_decomposer=add_decomposer,
        add_sql_planner=add_sql_planner,
        add_attribute_selector=add_attribute_selector,
        add_schema_cleaner=add_schema_cleaner,
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
@click.option("--use-table-schema", is_flag=True, default=False)
# Agent options
@click.option("--add-reask", is_flag=True, default=False)
@click.option("--add-empty-table", is_flag=True, default=False)
@click.option("--add-decomposer", is_flag=True, default=False)
@click.option("--add-sql-planner", is_flag=True, default=False)
@click.option("--add-attribute-selector", is_flag=True, default=False)
@click.option("--add-schema-cleaner", is_flag=True, default=False)
# Model options
@click.option("--max-tokens", type=int, default=1000)
@click.option("--temperature", type=float, default=0)
# Client options
@click.option("--api-provider", type=str, default="anthropic")
@click.option("--planner-api-provider", type=str, default="anthropic")
@click.option("--model", type=str, default="claude-3-opus-20240229")
@click.option("--planner-model", type=str, default="claude-3-opus-20240229")
@click.option("--api-key", type=str, required=True)
@click.option("--planner-api-key", type=str, required=True)
@click.option(
    "--client-cache-path",
    type=str,
    default="/home/lorr1/projects/code/meadow/test_cache.duckdb",
)
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
    use_table_schema: bool,
    add_reask: bool,
    add_empty_table: bool,
    add_decomposer: bool,
    add_sql_planner: bool,
    add_attribute_selector: bool,
    add_schema_cleaner: bool,
    max_tokens: int,
    temperature: float,
    api_provider: str,
    planner_api_provider: str,
    model: str,
    planner_model: str,
    client_cache_path: str,
    api_key: str,
    planner_api_key: str,
    overwrite_cache: bool,
) -> None:
    """Predict SQL."""
    random.seed(0)
    np.random.seed(0)

    locals_dict = locals()
    console.print(json.dumps(locals_dict, indent=2))

    if "custom" in dataset_path:
        benchmark = "custom"
    else:
        benchmark = "spider"

    # load client
    llm_config = LLMConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=0,
    )
    cache = DuckDBCache(client_cache_path)
    if api_provider == "anthropic":
        api_client = AnthropicClient(api_key=api_key)
    elif api_provider == "openai":
        api_client = OpenAIClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown api provider {api_provider}")

    if planner_api_provider == "anthropic":
        planner_api_client = AnthropicClient(api_key=planner_api_key)
    elif planner_api_provider == "openai":
        planner_api_client = OpenAIClient(api_key=planner_api_key)
    else:
        raise ValueError(f"Unknown api provider {planner_api_provider}")
    client = Client(
        cache=cache,
        api_client=api_client,
        model=model,
    )
    planner_client = Client(
        cache=cache,
        api_client=planner_api_client,
        model=planner_model,
    )

    console.print("Loading metadata...")
    db_to_tables = read_tables_json(table_meta_path, database_path, lowercase=lowercase_schema)

    console.print("Loading data...")
    data = load_data(dataset_path)
    if num_run > 0:
        console.print(f"Running on {min(len(data), num_run)} examples")
        data = data[:num_run]

    # data = [t for t in data if "flight" in t["db_id"]]
    original_data = data

    # load the examples
    console.print("Formatting data...")
    num_print = min(num_print, len(data))
    text_to_sql_in = [
        get_text_to_sql_in(
            benchmark=benchmark,
            input_question=input_question,
            db_to_tables=db_to_tables,
            database_path=Path(database_path),
            add_reask=add_reask,
            add_empty_table=add_empty_table,
            add_decomposer=add_decomposer,
            add_sql_planner=add_sql_planner,
            add_attribute_selector=add_attribute_selector,
            add_schema_cleaner=add_schema_cleaner,
        )
        for input_question in data
    ]

    # Get output filename
    full_dataset_path = Path(dataset_path)
    # Get todays date
    date_today = datetime.datetime.now().strftime("%y-%m-%d")
    if run_name:
        run_name = f"{run_name}_"

    agent_type_str = (
        ("reask_" if add_reask else "") +
        ("empty_" if add_empty_table else "") +
        ("decomposer_" if add_decomposer else "") +
        ("sqlplanner_" if add_sql_planner else "") +
        ("attribute_" if add_attribute_selector else "") +
        ("cleaner_" if add_schema_cleaner else "")
    )
    agent_type_str = agent_type_str.rstrip("_")
    suffix = f"{run_name}{full_dataset_path.stem}_{date_today}.json"  # noqa: E501
    prefix = f"{agent_type_str}_{model}_{planner_model}"
    output_filename = f"{prefix}_{suffix}"
    console.print(f"Saving to {Path(output_dir) / output_filename}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / output_filename, "w") as fout:
        for i in range(num_print):
            # Run a few to get some examples to console.print
            predictions, prompts = generate_sql(
                text_to_sql_in=[text_to_sql_in[i]],
                client=client,
                planner_client=planner_client,
                llm_config=llm_config,
                real_tables=db_to_tables if use_table_schema else None,
                overwrite_cache=overwrite_cache,
                async_batch_size=async_batch_size,
            )
            for prediction, prompt in zip(predictions, prompts):
                sql, tbl = prediction
                prediction = re.sub(r"[\s\t\n]+", " ", sql)
                console.print(f"[blue]Prompt:[/blue] {json.dumps(prompt, indent=2)}")
                console.print(f"[red]Prediction SQL:[/red] {sql}")
                console.print(f"[red]Prediction Table:[/red] {tbl}")
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
            planner_client=planner_client,
            llm_config=llm_config,
            real_tables=db_to_tables if use_table_schema else None,
            overwrite_cache=overwrite_cache,
            async_batch_size=async_batch_size,
        )

        for i in range(len(generated_sqls)):
            sql, tbl = generated_sqls[i]
            prompt = [m.laurel_serialize() if m else None for m in generated_prompts[i]]
            entry = {
                **original_data[i],
                "pred": sql,
                "pred_table": tbl,
                "prompt": prompt,
            }
            print(json.dumps(entry), file=fout)
    console.print(f"Saved to {Path(output_dir) / output_filename}")


if __name__ == "__main__":
    cli()
