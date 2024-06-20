"""Controller agent."""

import logging

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMPlannerAgent
from meadow.agent.planner import PlannerAgent, parse_replacements_in_instruction
from meadow.agent.schema import AgentMessage, ClientMessageRole, Commands
from meadow.agent.utils import print_message
from meadow.database.database import Database
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


class ControllerAgent(Agent):
    """The controller agent that manages the flow between other agents."""

    def __init__(
        self,
        supervisor: Agent,
        planner: LLMPlannerAgent,
        database: Database,
        name: str = "Controller",
        description: str = "Controller agent that manages the flow between other agents.",
        supervisor_auto_respond: bool = False,  # supervisor auto <next>
        can_solidify_intermediate_table_states: bool = True,  # whether to finalize drafts - False with executor convos
        silent: bool = True,
    ):
        self._supervisor = supervisor
        self._planner = planner
        self._database = database
        self._messages = MessageHistory()
        self._agent_executors: dict[Agent, list[ExecutorAgent]] = {
            agent: agent.executors
            for agent in planner.available_agents.values()
            if agent.executors
        }
        if planner.executors:
            self._agent_executors[planner] = planner.executors
        self._name = name
        self._description = description
        self._can_solidify_intermediate_table_states = (
            can_solidify_intermediate_table_states
        )
        # Current agent we are actively talking to (can be supervisor)
        self._current_agent: Agent = self._planner
        # Current agent solving the task (not the supervisor)
        self._current_task_agent: Agent = self._planner
        self._supervisor_auto_respond = supervisor_auto_respond
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
    def database(self) -> Database:
        """The database used by the agent."""
        return self._database

    def set_current_agent(self, agent: Agent) -> None:
        """Set the current task agent."""
        self._current_agent = agent

    def set_current_task_agent(self, agent: Agent) -> None:
        """Set the current task."""
        self._current_task_agent = agent

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
            logger.error("GOT EMPTY MESSAGE")
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
        if reply.is_termination_message:
            return
        if (
            not reply.requires_response
            and self._supervisor_auto_respond
            and self._current_agent == self._supervisor
        ):
            auto_response = AgentMessage(
                content=Commands.NEXT,
                sending_agent=self._supervisor.name,
                receiving_agent=self.name,
            )
            self._messages.add_message(
                agent=self._supervisor,
                agent_role=ClientMessageRole.SENDER,
                message=reply,
            )
            await self.receive(auto_response, self._supervisor)
        else:
            await self.send(reply, self._current_agent)

    async def _generate_next_step_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> AgentMessage:
        """Generate a reply to move to the next step in the task plan."""
        next_task = self._planner.move_to_next_agent()
        if next_task:
            self.set_current_task_agent(next_task.agent)
            self.set_current_agent(next_task.agent)
            next_message = next_task.prompt
            # The planner allows for instruction to have message replacment tags like {stepXX}
            # In the planner, we parse all the {stepXX} to be {agentName} so that right now
            # we can parse any {agentName} to be the last message from that agent. As executors
            # are directly processing outputs from agents, we do not want to change the input at all.
            # This  is expecially true for the planner parsing executor that needs to raw {stepXX}.
            if not isinstance(self._current_agent, ExecutorAgent):
                for last_agent_name_message in parse_replacements_in_instruction(
                    next_message
                ):
                    last_content = self._messages.get_messages(
                        self._planner.available_agents[last_agent_name_message]
                    )[-1].content
                    next_message = next_message.replace(
                        f"{{{last_agent_name_message}}}", last_content
                    )
            # Agents can be single LLMAgents or PlannerAgents. PlannerAgents own collections of subagents that they
            # coordinate between. We need to start a subchat with that planner agent here.
            if isinstance(self._current_agent, LLMPlannerAgent):
                # Generate a new chat between sender and executor
                sub_controller = ControllerAgent(
                    name=f"{self._supervisor.name}_{self._current_agent.name}_Controller",
                    supervisor=self._supervisor,
                    planner=self._current_agent,
                    database=self.database,
                    supervisor_auto_respond=True,
                    silent=self._silent,
                )
                self._supervisor.set_chat_role(AgentRole.SUPERVISOR)
                executor_response = await sub_controller.initiate_chat(next_message)
                self._supervisor.set_chat_role(AgentRole.TASK_HANDLER)

                next_message = executor_response.content
                next_display = executor_response.display_content
                # We just pretended we started and completed a chat with the task agent.
                # The conversation is now over, so we need to set the current agent to
                # the supervisor and add the final message to this controller's message history.
                self._messages.add_message(
                    agent=self._current_agent,
                    agent_role=ClientMessageRole.RECEIVER,
                    message=executor_response,
                )
                self.set_current_agent(self._supervisor)
            else:
                next_display = None
            is_termination_message = False
            assert next_message, "Prompt cannot be empty"
        else:
            self.set_current_agent(self._supervisor)
            next_message = messages[-1].content
            next_display = messages[-1].display_content
            is_termination_message = True
            # Unhide any tables as it's the end of a "task"
            if self._can_solidify_intermediate_table_states:
                self.database.unhide_all_tables()
        return AgentMessage(
            content=next_message,
            display_content=next_display,
            sending_agent=self.name,
            is_termination_message=is_termination_message,
        )

    async def _generate_executed_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> AgentMessage:
        """Generate a reply after execution.

        Some agents require executors - models to go and parse their responses
        and catch errors. If the message requires execution and the agent has executors,
        we start a hierarchical chat between the agent and its current executor.

        We send the result message (or default one if no executors) to either the
        supervisor or agent, depending on who send the message.
        """
        executor_response = None
        default_response = AgentMessage(
            content=messages[-1].content,
            display_content=messages[-1].display_content,
            sending_agent=self.name,
        )
        last_message = messages[-1]
        if messages[-1].requires_execution and self._agent_executors.get(sender, []):
            for executor in self._agent_executors.get(sender, []):
                # Generate a new chat between sender and executor
                sub_controller = ControllerAgent(
                    name=f"{sender.name}_{executor.name}_Controller",
                    supervisor=sender,
                    # Very simply planner that always sends instruction to executor
                    # in a single step.
                    planner=PlannerAgent(
                        available_agents=[executor],
                        client=None,
                        llm_config=None,
                        database=None,
                    ),
                    database=self.database,
                    can_solidify_intermediate_table_states=False,
                    supervisor_auto_respond=True,
                    silent=self._silent,
                )
                sender.set_chat_role(AgentRole.SUPERVISOR)
                # Warmstart the conversation with the sub_controller and sender with that
                # chat between self and sender
                sender.add_to_messages(sub_controller, sender.get_messages(self))
                sub_controller.add_to_messages(sender, self.get_messages(sender))
                executor_response = await sub_controller.initiate_chat(
                    last_message.content
                )
                last_message = executor_response.model_copy()
                sender.set_chat_role(AgentRole.TASK_HANDLER)
            # Finalize any temporary draft data views made and edited by the executors into final
            # views for the next step
            if self._can_solidify_intermediate_table_states:
                self.database.finalize_draft_views()
            # Add the last message from the sender with the final response. This is needed if we
            # meed to retrieve the final response from sender to use later on in the plan.
            self._messages.add_message(
                agent=sender,
                agent_role=ClientMessageRole.RECEIVER,
                message=last_message,
            )
            # Adding subcontroller message history in case need to query it later
            self.add_to_messages(sub_controller, sub_controller.get_messages(sender))
        final_response = last_message or default_response
        # The current_task_agent is the agent of the task that is currently answering the user's
        # question. current_agent is whoever we last chatted with. When a user responds, current_agent
        # will be the user and current_task_agent will be the llm agent of the task
        self.set_current_agent(
            self._supervisor if sender != self._supervisor else self._current_task_agent
        )
        return final_response

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> AgentMessage:
        """Generate a reply based on the received messages.

        We handle the DSL commands here and generate the appropriate reply. If the message
        is a termination message, we return it immediately. If the message is a next message,
        we move to the next step in the task plan. Otherwise, we generate the response and return.
        """
        if Commands.has_next(messages[-1].content):
            assert sender == self._supervisor, "Only user can send move on message."
            return await self._generate_next_step_reply(messages, sender)
        elif (
            Commands.has_end(messages[-1].content)
            # if not has_end but is termination, means the content is important
            # to be passed back to user but the message is terminating
            or messages[-1].is_termination_message
        ):
            # If user wants to quit, return to them and quit immediately
            # But if a agent says to end, that means move on to next step
            if sender == self._supervisor:
                self.set_current_agent(self._supervisor)
                return AgentMessage(
                    content=messages[-1].content,
                    display_content=messages[-1].display_content,
                    sending_agent=self.name,
                    is_termination_message=True,
                )
            else:
                return await self._generate_next_step_reply(messages, sender)
        else:
            return await self._generate_executed_reply(messages, sender)

    async def initiate_chat(
        self,
        input: str,
    ) -> AgentMessage | None:
        """Supervisor initiating a chat.

        Will return the last non termination message sent to the supervisor.
        """
        message = AgentMessage(
            agent_role=ClientMessageRole.RECEIVER,
            content=input,
            sending_agent=self._supervisor.name,
            receiving_agent=self.name,
        )
        await self.receive(message, self._supervisor)
        # The last message send to the controller that isn't a NEXT or END
        # message is the final response of a chat.
        all_messages = self._messages.get_messages_linearly_by_time()
        for message in reversed(all_messages):
            if not Commands.has_end(message.content) and not Commands.has_next(
                message.content
            ):
                return message
        return None
