import logging

from pydantic import BaseModel

from meadow.agent.agent import Agent
from meadow.agent.controller import ControllerAgent
from meadow.agent.data_agents.text2sql import SQLGeneratorAgent
from meadow.agent.executor.contrib.empty_result_debugger import EmptyResultExecutor
from meadow.agent.executor.contrib.sql_validate_reask import SQLValidateExecutor
from meadow.agent.planner import PlannerAgent
from meadow.agent.user import UserAgent
from meadow.client.client import Client
from meadow.client.schema import ChatMessage, ChatResponse, LLMConfig
from meadow.database.database import Database

logger = logging.getLogger(__name__)

SQL_PROMPT_WITH_PLANNER = """You generate SQLite SQL queries and are a SQLite expert. Given the table schema and user's question, generate a SQLite SQL query that answers the user's question and a one sentence description of the generated SQL query. Follow these rules:

1. Feel free to think through what you need to do first.
2. First use <description></description> tags for a one sentence description of what the table captures. Be concise.
3. Then <sql></sql> tags or ```sql...``` for the SQL. Please refer to views and base tables in the SQL if necessary.
4. Please use `FROM sqlXXX` to refer to the SQL query number XXX in the prompt. For example, if sql2 is in the schema from a prior step, please use `FROM sql2` to refer to that query.

The user's schema is:
{schema}"""

# SQL_PLANNER_PROMPT = """The user wants to answer an analytics question in SQL based on the following schema.

# {serialized_schema}
# The user's question may be complex or vague. Based on the following question provided by the user, please make a plan consisting of a sequence of sub-SQL-queries for how to answer this question using the following agents:

# <agents>
# {agents}
# </agents>

# For each sub-step in the sequence, indicate which agents should perform the task and generate a detailed instruction for the agent to follow. To reference a past step (e.g. step 2), please use the phrase `sql 2` to reference the second sql generated. You can use the same agent multiple times. If you are confused by the task or need more details, please ask for feedback. The user may also provide suggestions to the plan that you should take into account when generating the plan. When generating a plan, please use the following tag format to specify the plan.

# <steps>
# <step1>
# <agent>...</agent>
# <instruction>...</instruction>
# </step1>
# <step2>
# ...
# </step2>
# ...
# </steps>

# If the user responds back at some point with a message that indicates the user is satisfied with the plan, ONLY output {termination_message} tags to signal an end to the conversation. {termination_message} tags should only be used in isolation of all other tags.

# Lastly, use as few steps as possible."""


SQL_PLANNER_PROMPT_SPIDER = """The user wants to answer an analytics question in SQL.

Based on the following question provided by the user, please make a plan consisting of a minimal sequence of SQL-queries using the following agent:

{agents}

For each SQL in the sequence, indicate which agents should perform the task and generate an instruction for the agent to follow using prior steps. To reference a past step (e.g. to reference step 2, please use the phrase `sql2` in the instruction). You can use the same agent multiple times. When generating a plan, please use the following tag format to specify the plan.

<steps>
<step1>
<agent>...</agent>
<instruction>...</instruction>
</step1>
<step2>
...
</step2>
...
</steps>

The user's schema is below.

{serialized_schema}

Please keep the plan simple and use as few steps as possible. Be very explicit about which columns should be selected at the end. Only select exactly what the user asks and nothing more."""


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
    planner_client: Client,
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
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=[
            SQLValidateExecutor(
                client=None,
                llm_config=llm_config,
                database=database,
                max_execution_attempts=0,
            ),
        ],
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=None,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller


def get_text2sql_planner_agent(
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
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
        executors=[
            SQLValidateExecutor(
                client=None,
                llm_config=llm_config,
                database=database,
                max_execution_attempts=0,
            ),
        ],
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=planner_client,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_planner,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller


def get_text2sql_simple_reask_agent(
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
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
    # Have to build custom executor that doesn't use LLM
    no_llm_executor = [
        SQLValidateExecutor(
            client=None,
            llm_config=None,
            database=database,
        ),
    ]
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=no_llm_executor,
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=None,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller


def get_text2sql_simple_reask_planner_agent(
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
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
    # Have to build custom executor that doesn't use LLM
    no_llm_executor = [
        SQLValidateExecutor(
            client=None,
            llm_config=None,
            database=database,
        ),
    ]
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=no_llm_executor,
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=planner_client,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_planner,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller


def get_text2sql_llm_reask_agent(
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
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
    callback_empty = lambda model_messages, chat_response: model_callback(
        model_messages,
        chat_response,
        example_idx,
        "EmptyResultExecutor",
        all_prompts_to_save,
    )
    llm_executor = [
        SQLValidateExecutor(
            client=None,
            llm_config=None,
            database=database,
        ),
        EmptyResultExecutor(
            client=client,
            llm_config=llm_config,
            database=database,
            llm_callback=callback_empty,
        )
    ]
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=llm_executor,
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=None,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller


def get_text2sql_llm_reask_planner_agent(
    user_agent: UserAgent,
    client: Client,
    planner_client: Client,
    llm_config: LLMConfig,
    database: Database,
    overwrite_cache: bool,
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
    callback_empty = lambda model_messages, chat_response: model_callback(
        model_messages,
        chat_response,
        example_idx,
        "EmptyResultExecutor",
        all_prompts_to_save,
    )
    callback_planner = lambda model_messages, chat_response: model_callback(
        model_messages, chat_response, example_idx, "PlannerAgent", all_prompts_to_save
    )
    llm_executor = [
        SQLValidateExecutor(
            client=None,
            llm_config=None,
            database=database,
        ),
        EmptyResultExecutor(
            client=client,
            llm_config=llm_config,
            database=database,
            llm_callback=callback_empty,
        )
    ]
    text2sql = SQLGeneratorAgent(
        client=client,
        llm_config=llm_config,
        database=database,
        executors=llm_executor,
        system_prompt=SQL_PROMPT_WITH_PLANNER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_sql,
    )
    planner = PlannerAgent(
        available_agents=[text2sql],
        client=planner_client,
        llm_config=llm_config,
        database=database,
        system_prompt=SQL_PLANNER_PROMPT_SPIDER,
        overwrite_cache=overwrite_cache,
        llm_callback=callback_planner,
    )
    controller = ControllerAgent(supervisor=user_agent, planner=planner, database=database, silent=True)
    return controller
