from typing import Callable

from meadow.agent.data_agents.text2sql_utils import check_empty_table
from meadow.agent.executor.debugger import DebuggerExecutor
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database

DEFAULT_EMPTYRESULT_PROMPT = """You job is to help debug why a given SQL query is returning an empty table result. There's a chance nothing is wrong and it's just an empty result, but there's also a chance the query is incorrect. Common issues are incorrect WHERE clauses where the filter is wrong, incorrect use of logical operators (OR vs AND), incorrect handling of NULL values, and incorrect joins.

The user will provide their schema and associated empty SQL query. You have three possible actions you can take to resolve the issue

1. [Query] Issue a single SELECT SQL query on the database of any kind.
2. [Edit] Modify the SQL.
3. [Do Nothing] If you believe the SQL is correct and the table is appropriately empty.

You must always output one action of the following form:

---
Action: Query/Edit/Do Nothing
Input: ```Input to action in quotes```
---

Often, a good first step is to SELECT * ... LIMIT 100 from a table to see all the values and attributes in the table. Incorrect filters or values can be identified this way, but you may need to be more creative if the table is very large.

The user will provide the response and you two will collectively iterate on the issue until it is resolved. Your final goal is to edit the SQL to be correct or do nothing. If you would like, please plan the action to take before outputting the action in the format above."""


class EmptyResultExecutor(DebuggerExecutor):
    """Debugger executor specifically for empty tables."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        max_execution_attempts: int = 2,
        system_prompt: str = DEFAULT_EMPTYRESULT_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the debugger executor."""
        super().__init__(
            client=client,
            llm_config=llm_config,
            database=database,
            max_execution_attempts=max_execution_attempts,
            execution_func=check_empty_table,
            system_prompt=system_prompt,
            overwrite_cache=overwrite_cache,
            silent=silent,
            llm_callback=llm_callback,
        )
