from typing import Callable

from meadow.agent.data_agents.text2sql_utils import check_empty_table
from meadow.agent.executor.debugger import DebuggerExecutor
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database

DEFAULT_EMPTYRESULT_PROMPT = """You job is to help debug why a given SQL query is returning an empty table result. There's a chance nothing is wrong and it's just an empty result, but there's also a chance the query is incorrect. Common issues are

* Incorrect WHERE clauses where the filter is wrong. The query or question may have a misspelling, incomplete value, wrong punctuation, or other issues.
* Incorrect use of logical operators (OR vs AND). The booleans may not match what the user is asking.
* Incorrect handling of NULL values. The filter may be incorrectly adding or removing NULL values.
* Incorrect joins where the join keys are not matching. Not all dependencies are added to the schema so the joins may need to be debugged.
* Incorrect CTE expressions. A subtable may be empty, causing the downstream query to be empty.

The user will provide their schema and associated empty SQL query. You have three possible actions you can take to resolve the issue

1. [Query] Issue a single SELECT SQL query on the database of any kind.
2. [Edit] Modify the SQL.
3. [Do Nothing] If you believe the SQL is correct and the table is appropriately empty.

You must always output one action of the following form:

---
Action: Query/Edit/Do Nothing
Input: ```Input to action in quotes```
---

Your final goal is to edit the SQL to be correct or do nothing. Often, a good first step is to Query with 'SELECT <relevant_attributes> FROM <table> LIMIT 100' without a filter to see all the values and attributes in the table. If the query has a lot of CTEs, you may need to start debugging them one at a time."""

DEFAULT_EMPTYRESULT_DESC = (
    "Debugs why a table result is empty by querying and exploring the database."
)


class EmptyResultExecutor(DebuggerExecutor):
    """Debugger executor specifically for empty tables."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        max_execution_attempts: int = 2,
        description: str = DEFAULT_EMPTYRESULT_DESC,
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
            description=description,
            system_prompt=system_prompt,
            overwrite_cache=overwrite_cache,
            silent=silent,
            llm_callback=llm_callback,
        )
