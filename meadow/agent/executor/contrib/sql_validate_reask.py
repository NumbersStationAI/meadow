from typing import Callable

from meadow.agent.data_agents.text2sql_utils import parse_sql_response
from meadow.agent.executor.reask import ReaskExecutor
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database

DEFAULT_SQL_EXECUTOR_PROMPT = """You are a error debugging SQL assistant who needs to help a user understand their SQL error message and suggest how to fix it.

Below is the data schema the user is working with.
{serialized_schema}

Given the user's message below, please explain the error and suggset in freeform text how to fix the query. The query may need additional tables, fixes in function use, or other modifications. Please provide a detailed explanation of the error and how to fix it."""


class SQLValidateExecutor(ReaskExecutor):
    """Agent that execute/validates a response given an execution function."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        max_execution_attempts: int = 2,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the sql validate reask agent."""
        super().__init__(
            client=client,
            llm_config=llm_config,
            database=database,
            max_execution_attempts=max_execution_attempts,
            execution_func=parse_sql_response,
            system_prompt=DEFAULT_SQL_EXECUTOR_PROMPT,
            overwrite_cache=overwrite_cache,
            silent=silent,
            llm_callback=llm_callback,
        )
