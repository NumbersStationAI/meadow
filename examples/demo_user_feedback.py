import argparse
import asyncio

from meadow import Client
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.attribute_detector import AttributeDetectorAgent
from meadow.agent.data_agents.basic import BasicAgent
from meadow.agent.data_agents.column_generation import ColumnGeneratorAgent
from meadow.agent.data_agents.planner_constraints import (
    attribute_detector_constraints,
    column_generator_constraints,
    metadata_question_constraints,
    question_refiner_constraints,
    sql_agent_constraints,
)
from meadow.agent.data_agents.schema_renamer import SchemaRenamerAgent
from meadow.agent.data_agents.table_selector import SingleTableSelectorAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.executor.user_feedback import UserFeedbackExecutor
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, ExecutorFunctionInput
from meadow.agent.user import UserAgent
from meadow.agent.utils import print_message
from meadow.cache import DuckDBCache
from meadow.client.api.anthropic import AnthropicClient
from meadow.client.api.api_client import APIClient
from meadow.client.api.openai import OpenAIClient
from meadow.client.api.samba import SambaClient
from meadow.client.api.together import TogetherClient
from meadow.client.schema import LLMConfig
from meadow.database.connector.connector import Connector
from meadow.database.connector.duckdb import DuckDBConnector
from meadow.database.connector.sqlite import SQLiteConnector
from meadow.database.database import Database


def get_text2sql_agent(
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
    text2data_task = ColumnGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    table_selector = SingleTableSelectorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    column_gen_planner = PlannerAgent(
        available_agents=[table_selector, text2data_task],
        client=big_client,
        llm_config=llm_config,
        database=database,
        constraints=[
            column_generator_constraints,
        ],
        name="ColumnGenerator",
        description="This agent augments a user's data with a new column. New columns are useful if the user is requesting some attribute that doesn't exist or there is lacking a join column to join tables together.",
        overwrite_cache=overwrite_cache,
    )
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        overwrite_cache=overwrite_cache,
    )
    text_to_sql_planner = PlannerAgent(
        available_agents=[schema_cleaner, attribute_detector, text2sql],
        client=big_client,
        llm_config=llm_config,
        database=database,
        constraints=[
            attribute_detector_constraints,
            sql_agent_constraints,
        ],
        name="SQLGenerator",
        description="This agent generates SQL given a user's question. When using this agent, make sure the <instruction> is the user's text question or the output from another step. Do NOT give this agent SQL. This agent should only be called once.",
        overwrite_cache=overwrite_cache,
    )

    def parse_for_user_feedback(
        input: ExecutorFunctionInput,
    ) -> AgentMessage:
        content = input.messages[-1].content
        if "<clear>" in content:
            inside_clear = content.split("<clear>")[1].split("</clear>")[0].strip("[]")
            return AgentMessage(
                content=inside_clear,
                display_content=f"The question we will ask the model is: {inside_clear}.",
                sending_agent=input.agent_name,
                requires_response=False,
            )
        else:
            if "<feedback>" in content:
                inside_feedback = (
                    content.split("<feedback>")[1].split("</feedback>")[0].strip("[]")
                )
            else:
                inside_feedback = content
            return AgentMessage(
                content=inside_feedback,
                sending_agent=input.agent_name,
                requires_response=True,
            )

    unclear_question = BasicAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=[
            UserFeedbackExecutor(
                client=client,
                llm_config=llm_config,
                database=database,
                execution_func=parse_for_user_feedback,
                overwrite_cache=overwrite_cache,
            ),
        ],
        name="UnclearQuestion",
        description="Checks if the user's question is unclear in that is it high level or mentions unclear dimensions or metrics. The instruction to this agent must be only the user question in entirety so it can maintain context. The output of this agent should be passed through with {stepXX} tags.",
        system_prompt="""You are a helpful analyst assistant who is making sure the user is asking good questions. Given a user question, please determine if the question is clear and if not, ask for clarification on the definitions or provide three suggested questions to help the user refine their question. Make sure all your responses are grounded in the schema and actually answerable. You will output the tag <clear>[final question]</clear> for a clear question that contains all the context you learned from the user: question and definitions. If you need feedback, please respond with <feedback>[clarifying question or question suggestions]</feedback>.

The user's schema is:

{schema}


Please respond with <clear> or <feedback>.""",
    )

    metadataset_question = BasicAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        name="MetadataQuestion",
        description="Answers a user's question about a dataset that does not directly map to SQL. E.g., what are some tables, explain how the data works, or what is the data about. This agent should be used alone with no other.",
        system_prompt="""You are a helpful analyst assistant who is answering meta questions about the dataset.

The user's schema is:

{schema}

Do not output any code or SQL. Just answer the question about the dataset in freeform text. Make sure to stay grounded in the schema and be faithful to the data values and names.""",
    )

    vague_planner = PlannerAgent(
        available_agents=[unclear_question],
        client=big_client,
        llm_config=llm_config,
        database=database,
        name="QuestionRefiner",
        description="If the user's question can be answered with a SQL query, do NOT use this agent. If the question is vague, call this agent to clarify the question. The instruction to this agent must be only the user question in entirety so it can maintain context. If this agent is used, it _must_ be followed with SQL generator agent. The output of this agent should be passed to the SQL generator via {stepXX} tags.",
        overwrite_cache=overwrite_cache,
    )

    final_planner = PlannerAgent(
        available_agents=[
            metadataset_question,
            vague_planner,
            column_gen_planner,
            text_to_sql_planner,
        ],
        client=big_client,
        llm_config=llm_config,
        database=database,
        constraints=[
            attribute_detector_constraints,
            sql_agent_constraints,
            question_refiner_constraints,
            metadata_question_constraints,
        ],
        overwrite_cache=overwrite_cache,
    )
    controller = ControllerAgent(
        supervisor=user_agent,
        planner=final_planner,
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
    elif api_provider == "together":
        api_client = TogetherClient(api_key=api_key)
    elif api_provider == "samba":
        api_client = SambaClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")
    client = Client(
        cache=cache,
        api_client=api_client,
        model=model,
    )
    big_client = Client(cache=cache, api_client=api_client, model=model)
    llm_config = LLMConfig(
        max_tokens=1000,
        temperature=0.0,
    )
    user = UserAgent(name="User", silent=False)
    controller = get_text2sql_agent(
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
        help="Model.",
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
