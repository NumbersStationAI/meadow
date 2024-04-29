"""SQL Generator Agent."""

import logging
import re

import sqlglot

from meadow.agent.agent import Agent, LLMAgent
from meadow.agent.schema import AgentMessage
from meadow.agent.utils import (
    generate_llm_reply,
    has_termination_condition,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)

DEFAULT_SQL_PROMPT = """Given the table schema and user's question, generate a DuckDB SQL query that answers it. If the query involves multiple steps, describe each step in detail. Use <sql1>, <sql2>, ... tags for the SQL, depending on if previous queries were talked about in the conversation. IMPORTANT: if you want to use a prior query's result as a subquery or table, use the relevant <sql/> tag in query. Once the user is satisfied with the SQL, output {termination_message} tag.

<schema>
  <table name="orders">
    <column name="order_id" type="integer" primaryKey="true" autoIncrement="true"/>
    <column name="order_date" type="datetime"/>
    <column name="total_cost" type="decimal(10,2)"/>
    <column name="status" type="varchar(20)"/>
    <column name="customer_id" type="integer" foreignKey="true" references="customers(customer_id)"/>
    <column name="supplier_id" type="integer" foreignKey="true" references="suppliers(supplier_id)"/>
  </table>

  <table name="order_items">
    <column name="order_item_id" type="integer" primaryKey="true" autoIncrement="true"/>
    <column name="order_id" type="integer" foreignKey="true" references="orders(order_id)"/>
    <column name="product_id" type="integer" foreignKey="true" references="products(product_id)"/>
    <column name="quantity" type="integer"/>
    <column name="unit_price" type="decimal(10,2)"/>
  </table>

  <table name="customers">
    <column name="customer_id" type="integer" primaryKey="true" autoIncrement="true"/>
    <column name="first_name" type="varchar(50)"/>
    <column name="last_name" type="varchar(50)"/>
    <column name="email" type="varchar(100)"/>
    <column name="phone" type="varchar(20)"/>
    <column name="address" type="varchar(200)"/>
    <column name="city" type="varchar(50)"/>
    <column name="state" type="varchar(50)"/>
    <column name="zip_code" type="varchar(20)"/>
    <column name="country" type="varchar(50)"/>
  </table>

  <table name="products">
    <column name="product_id" type="integer" primaryKey="true" autoIncrement="true"/>
    <column name="name" type="varchar(100)"/>
    <column name="description" type="text"/>
    <column name="category" type="varchar(50)"/>
    <column name="unit_price" type="decimal(10,2)"/>
    <column name="units_in_stock" type="integer"/>
    <column name="supplier_id" type="integer" foreignKey="true" references="suppliers(supplier_id)"/>
  </table>

  <table name="suppliers">
    <column name="supplier_id" type="integer" primaryKey="true" autoIncrement="true"/>
    <column name="company_name" type="varchar(100)"/>
    <column name="contact_name" type="varchar(100)"/>
    <column name="email" type="varchar(100)"/>
    <column name="phone" type="varchar(20)"/>
    <column name="address" type="varchar(200)"/>
    <column name="city" type="varchar(50)"/>
    <column name="state" type="varchar(50)"/>
    <column name="zip_code" type="varchar(20)"/>
    <column name="country" type="varchar(50)"/>
  </table>
</schema>
"""


def parse_sqls(message: str) -> dict[str, str]:
    """Extract the SQL query from the response."""
    # find <sql1>, <sql2>, ... tags and return the SQL query
    # this will return a list of tuples where the tuples are the groups
    sql_components = re.findall(r"(<sql\d+>(.*?)<\/sql\d+>)", message, re.DOTALL)
    if not sql_components:
        raise ValueError("SQL not found in the response.")
    sql_dict = {}
    for sql_pair in sql_components:
        sql_with_tag, sql = sql_pair
        match = re.search(r"<sql(\d+)>", sql_with_tag)
        if match:
            sql.replace(";", "").strip()
            # try to parse and make look pretty
            try:
                sql = sqlglot.parse_one(sql, dialect="duckdb").sql(
                    dialect="duckdb", pretty=True
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse SQL in DuckDB format. sql={sql}, e={e}"
                )
                pass
            sql_dict[int(match.group(1))] = sql
    return sql_dict


def compile_subqueries(sql: str, sql_history: dict[str, str]) -> str:
    """Return SQL over base tables by replacing any subqueries from history."""

    for k, v in sql_history.items():
        if f"(<sql{k}>)" in sql:
            sql = sql.replace(f"<sql{k}>", f"{v}")
        else:
            sql = sql.replace(f"<sql{k}>", f"({v})")
    return sql


class SQLGeneratorAgent(LLMAgent):
    """Agent that generates SQL queries from user questions."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        system_prompt: str = DEFAULT_SQL_PROMPT,
        termination_message: str = "<exit>",
        overwriting_cache: bool = False,
        silent: bool = True,
    ):
        """Initialize the SQL generator agent."""
        self._client = client
        self._llm_config = llm_config
        self._system_prompt = system_prompt
        self._termination_message = termination_message
        self._overwriting_cache = overwriting_cache
        self._silent = silent
        self._messages = MessageHistory()
        self._sql_history: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "SQLGenerator"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Generates SQL queries based on user questions."

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        return self._system_prompt.format(termination_message=self._termination_message)

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            logger.error("GOT EMPTY MESSAGE")
            raise ValueError("Message is empty")
        self._messages.add_message(agent=recipient, role="assistant", message=message)
        await recipient.receive(message, self)

    async def receive(
        self,
        message: AgentMessage,
        sender: Agent,
    ) -> None:
        """Receive a message from another agent."""
        if not self._silent:
            print_message(
                message,
                from_agent=sender.name,
                to_agent=self.name,
            )
        self._messages.add_message(agent=sender, role="user", message=message)

        reply = await self.generate_reply(
            messages=self._messages.get_messages(sender), sender=sender
        )
        await self.send(reply, sender)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages."""
        chat_response = await generate_llm_reply(
            client=self.llm_client,
            messages=messages,
            tools=[],
            system_message=AgentMessage(
                role="system",
                content=self.system_message,
                generating_agent=self.name,
            ),
            llm_config=self._llm_config,
            overwrite_cache=self._overwriting_cache,
        )
        content = chat_response.choices[0].message.content
        print("SQL AGENT CONTENT", content)
        if has_termination_condition(content, self._termination_message):
            return AgentMessage(
                role="assistant",
                content=content,
                tool_calls=None,
                generating_agent=self.name,
                is_termination_message=True,
            )
        else:
            sql_dict = parse_sqls(content)
            # update history with new SQL
            for k, v in sql_dict.items():
                if k in self._sql_history:
                    if v != self._sql_history[k]:
                        print("BAD", k, self._sql_history[k], v)
                    continue
                # remove past <sql> references from the new SQL
                self._sql_history[k] = compile_subqueries(v, self._sql_history)
            # get the last sql and return it fully parsed
            last_sql = self._sql_history[max(self._sql_history.keys())]
            return AgentMessage(
                role="assistant",
                content=last_sql,
                tool_calls=None,
                generating_agent=self.name,
            )
