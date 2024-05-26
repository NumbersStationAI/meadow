"""Agent that outputs the user desired attributes from a question."""

import logging
from typing import Callable

from meadow.agent.agent import Agent, ExecutorAgent, LLMAgent
from meadow.agent.schema import AgentMessage, Commands
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

DEFAULT_ATTRIBUTE_PROMPT = """The user wants to answer an analytics question in SQL.

Below is the data schema the user is working with.
{schema}

Given the user's question, output exactly what are the final output attributes the user wants to see. They want the smallest set of attributes and prefer the most user friendly names (e.g. avoid ids and codes and attributes meant for joins).

For example, suppose the user is asking over a table with columns people(uid, name, last_name, age, country) and employment(eid, uid, cid, job, salary) and companies(cid, name, country).

Question: Who is the youngest person in the US?
Explanation: The user wants to know the person name from the people table of the person with the minimum age who is in the US country.
Attributes: people.name

Question: List the job and salary of the youngest employee?
Explanation: The user wants to know the employment job and employment salary of the person with the minimum age from the people table.
Attributes: employment.job, employment.salary

Question: What is the average salary of people in the US and what company do they work for?
Explanation: The user wants to know the average employment salary of people from the US country in the people table and the company name they work for.
Attributes: AVG(employment.salary), companies.name

Now output an explanation followed by the attributes the user wants to see for the following question:"""


class AttributeDetectorAgent(LLMAgent):
    """Agent that generates SQL queries from user questions."""

    def __init__(
        self,
        client: Client,
        llm_config: LLMConfig,
        database: Database,
        system_prompt: str = DEFAULT_ATTRIBUTE_PROMPT,
        overwrite_cache: bool = False,
        silent: bool = True,
        llm_callback: Callable = None,
    ):
        """Initialize the SQL generator agent."""
        self._client = client
        self._llm_config = llm_config
        self._database = database
        self._system_prompt = system_prompt
        self._overwrite_cache = overwrite_cache
        self._llm_callback = llm_callback
        self._silent = silent
        self._messages = MessageHistory()

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "AttributeDetector"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Augments the user question with desired attributes and outputs a new question to be used to generate the answer. The instruction to this agent should be a copy of the user question. The output of this agent should always be used downstream. Make the entire instruction be {stepXX} when using this agent's output."

    @property
    def llm_client(self) -> Client:
        """The LLM client of this agent."""
        return self._client

    @property
    def database(self) -> Database:
        """The database used by the agent."""
        return self._database

    @property
    def system_message(self) -> str:
        """Get the system message."""
        serialized_schema = serialize_as_list(self.database.tables)
        return self._system_prompt.format(
            schema=serialized_schema,
            termination_message=Commands.END,
            dialect="SQLite",
        )

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            logger.error("GOT EMPTY MESSAGE")
            raise ValueError("Message is empty")
        message.receiving_agent = recipient.name
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
        """Generate a reply when Executor agent."""
        chat_response = await generate_llm_reply(
            client=self.llm_client,
            messages=messages,
            tools=[],
            system_message=AgentMessage(
                role="system",
                content=self.system_message,
                sending_agent=self.name,
            ),
            llm_config=self._llm_config,
            llm_callback=self._llm_callback,
            overwrite_cache=self._overwrite_cache,
        )
        content = chat_response.choices[0].message.content
        content = content.split("Attributes:", 1)[-1].strip()
        return AgentMessage(
            role="assistant",
            content=messages[-1].content
            + f" The final attributes should be {content}.",
            sending_agent=self.name,
        )
