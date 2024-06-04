"""SQL decomposer agent."""

import logging
from queue import Queue
from typing import Callable

from meadow.agent.agent import (
    Agent,
    LLMAgent,
    SubTask,
)
from meadow.agent.schema import AgentMessage, ClientMessageRole
from meadow.agent.utils import (
    generate_llm_reply,
    print_message,
)
from meadow.client.client import Client
from meadow.client.schema import LLMConfig
from meadow.database.database import Database
from meadow.database.serializer import serialize_as_list
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


DEFAULT_SQL_PLAN_PROMPT = """You are an expert SQL teacher and explain to the user how to use the schema to answer the question.

Below is the data schema the user is working with.

{serialized_schema}

Explain the logical steps for how to answer the question in SQL and how to break it down into subqueries. For each step, give a name to each subquery, explain what it should answer, explain how it joins back with previous steps, and list any advanced operators (e.g. ROW_NUMBER, RANK, LAG) that need to be used. Make sure to end with the phrase `The final attributes should be` followed by the final attributes that the user wants to get from the database.

For example, suppose the user is asking over a table with columns people(uid, name, last_name, age, country) and employment(eid, uid, cid, job, salary) and companies(cid, name, country).

If the question is 'Find the top two companies by the number employees over the age of 50 and output their name', you should output

1. Find the employees over the age of 50.
    - Explanation: This will select employees from the people table and filter where the age is greater than 50.
    - Attributes: people.uid
    - CTE Name: WITH employees_over_50 AS (...)
    - Joins: None
2. Find the number of employees from `employees_over_50` who work for each company.
    - Explanation: This will count the number of employees from the employment table who work for each company.
    - Attributes: employment.cid, COUNT(employees_over_50.uid) AS num_employees
    - CTE Name: WITH employees_per_company AS (...)
    - Joins: `employees_over_50` JOIN employment ON employees_over_50.uid = employment.uid
3. Find the top two companies by the number of employees.
    - Explanation: This will rank the companies by the number of employees from `employees_per_company` and select the top two.
    - Attributes: companies.name
    - CTE Name: WITH top_companies AS (...)
    - Joins: `employees_per_company` JOIN companies ON employees_per_company.cid = companies.cid
    - Functions: Use a window function with ROW_NUMBER() OVER (PARTITION BY employees_per_company.cid ORDER BY num_employees DESC) AS rank
The final attributes should be companies.name."""


DEFAULT_SQL_PLAN_DESC = """This agent is expensive and takes as input a complex user question that often require numerous nested reasoning steps and outputs a plan to answer it. This agent should only be used sparingly for complex questions that have multiple steps and nested logic. The output of this agent **must** be used as input to another agent via {stepXX} tags."""


class SQLPlannerAgent(LLMAgent):
    """Agent that generates a plan for subsql tasks."""

    def __init__(
        self,
        client: Client | None,
        llm_config: LLMConfig | None,
        database: Database | None,
        name: str = "SQLPlanner",
        description: str = DEFAULT_SQL_PLAN_DESC,
        system_prompt: str = DEFAULT_SQL_PLAN_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the planner agent."""
        self._client = client
        self._llm_config = llm_config.model_copy()
        self._llm_config.max_tokens = 2000
        self._database = database
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._messages = MessageHistory()
        self._plan: Queue[SubTask] = Queue()
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def system_message(self) -> str:
        """Get the system message."""
        serialized_schema = serialize_as_list(self._database.tables)
        return self._system_prompt.format(
            serialized_schema=serialized_schema,
        )

    def get_messages(self, chat_agent: "Agent") -> list[AgentMessage]:
        """Get the messages between self and the chat_agent."""
        return self._messages.get_messages(chat_agent)

    def add_to_messages(
        self, chat_agent: "Agent", messages: list[AgentMessage]
    ) -> None:
        """Add chat messages between self and chat_agent.

        Used when starting hierarchical chats and historical messages
        need to be passed to the agent.
        """
        self._messages.copy_messages_from(chat_agent, messages)

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            raise ValueError("Message is empty")
        self._messages.add_message(
            agent=recipient, agent_role=ClientMessageRole.SENDER, message=message
        )
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
        self._messages.add_message(
            agent=sender, agent_role=ClientMessageRole.RECEIVER, message=message
        )

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
                agent_role=ClientMessageRole.SYSTEM,
                content=self.system_message,
                sending_agent=self.name,
            ),
            llm_config=self._llm_config,
            llm_callback=self._llm_callback,
            overwrite_cache=self._overwrite_cache,
        )
        content = chat_response.choices[0].message.content
        return AgentMessage(
            content=content,
            sending_agent=self.name,
        )
