"""Controller agent."""

import logging

from meadow.agent.agent import Agent, AgentRole, ExecutorAgent, LLMPlannerAgent
from meadow.agent.planner import PlannerAgent, parse_replacements_in_instruction
from meadow.agent.schema import AgentMessage, Commands
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
        supervisor_auto_respond: bool = False,
        silent: bool = True,
        name: str = "Controller",
        # LAUREL: JANKY
        solidify_drafts: bool = True,
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
        # Current agent we are actively talking to (can be supervisor)
        self._current_agent: Agent = self._planner
        # Current agent solving the task (not the supervisor)
        self._current_task_agent: Agent = self._planner
        self._supervisor_auto_respond = supervisor_auto_respond
        self._silent = silent
        self._name = name
        self._solidify_drafts = solidify_drafts

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Overlord"

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

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
    ) -> None:
        """Send a message to another agent."""
        if not message:
            logger.error("GOT EMPTY MESSAGE")
            raise ValueError("Message is empty")
        # print(
        #     colored(
        #         f"IN SEND {self.name} (assistant role) WITH RECIPIENT {recipient.name}",
        #         "red",
        #     )
        # )
        # print("MESSAGE", message.content, "\n\nDISPLAY", message.display_content)
        # print("-----")
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
        # update the message history
        # print(
        #     colored(
        #         f"IN RECIEVE {self.name} (user role) WITH SENDER {sender.name}", "blue"
        #     )
        # )
        # print("MESSAGE", message.content, "\n\nDISPLAY", message.display_content)
        # print("-----")
        self._messages.add_message(agent=sender, role="user", message=message)

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
                role="assistant",
                content=Commands.NEXT,
                sending_agent=self.name,
            )
            self._messages.add_message(
                agent=self._supervisor, role="assistant", message=reply
            )
            # self._supervisor._messages.add_message(
            #     agent=self, role="user", message=reply
            # )
            # self._supervisor._messages.add_message(
            #     agent=self, role="assistant", message=auto_response
            # )
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
            # The planner allows for instruction to have message replacment tags like {{stepXX}}
            # In the planner, we parse all the {{stepXX}} to be {{agentName}} so that right now
            # we can parse any {{agentName}} to be the last message from that agent.
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
                self._supervisor.set_chat_role(AgentRole.EXECUTOR)

                next_message = executor_response.content
                next_display = executor_response.display_content
                # We have finished the chat with the task agent so go back to supervisor
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
        return AgentMessage(
            role="assistant",
            content=next_message,
            display_content=next_display,
            sending_agent=self.name,
            is_termination_message=is_termination_message,
        )

    async def _generate_executed_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> AgentMessage:
        """Generate a reply after execution.

        FOX ME
        """
        executor_response = None
        default_response = AgentMessage(
            role="assistant",
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
                    supervisor_auto_respond=True,
                    silent=self._silent,
                    # SUPER JANKY
                    solidify_drafts=False,
                )
                sender.set_chat_role(AgentRole.SUPERVISOR)
                # TODO: clean up
                sender._messages._history[sub_controller] = [
                    msg.model_copy() for msg in sender._messages._history[self]
                ]
                sub_controller._messages._history[sender] = [
                    msg.model_copy() for msg in self._messages._history[sender]
                ]
                executor_response = await sub_controller.initiate_chat(
                    last_message.content
                )
                last_message = executor_response.model_copy()
                sender.set_chat_role(AgentRole.EXECUTOR)
            # Finalize any temporary draft data views made and edited by the executors into final
            # views for the next step
            if self._solidify_drafts:
                self.database.finalize_draft_views()
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
        """Generate a reply based on the received messages."""
        # TODO: add comments once finalized
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
                    role="assistant",
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
            role="user", content=input, sending_agent=self._supervisor.name
        )
        await self.receive(message, self._supervisor)
        all_messages = self._messages.get_messages_linearly_by_time()
        print("CHAT RESPONSE", self.name, "SUP", self._supervisor.name)
        for message in reversed(all_messages):
            if not Commands.has_end(message.content) and not Commands.has_next(
                message.content
            ):
                print(
                    "RETURNING", message.content, "-----\n\n", message.display_content
                )
                return message
        return None
