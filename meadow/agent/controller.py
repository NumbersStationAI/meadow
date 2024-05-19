"""Controller agent."""

import logging

from termcolor import colored

from meadow.agent.agent import Agent, AgentRole
from meadow.agent.exectors.reask import ExecutorAgent
from meadow.agent.planner import PlannerAgent
from meadow.agent.schema import AgentMessage, Commands
from meadow.agent.utils import print_message
from meadow.history.message_history import MessageHistory

logger = logging.getLogger(__name__)


class ControllerAgent(Agent):
    """The controller agent that manages the flow between other agents."""

    def __init__(
        self,
        supervisor: Agent,
        planner: PlannerAgent,
        supervisor_auto_respond: bool = False,
        silent: bool = True,
    ):
        self._supervisor = supervisor
        self._planner = planner
        self._messages = MessageHistory()
        self._agent_executors: dict[Agent, list[ExecutorAgent]] = {
            agent: agent.executors
            for agent in planner._available_agents.values()
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

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return "Controller"

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Overlord"

    def set_current_agent(self, agent: Agent) -> None:
        """Set the current task agent."""
        print("SWAPPING", agent)
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
        print(colored(f"IN SEND ADDING ASSISTANT {recipient.name}", "red"))
        print("MESSAGE", message.content)
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
        print(colored(f"IN RECIEVE ADDING USER {sender.name}", "blue"))
        print("MESSAGE", message.content)
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
                generating_agent=self.name,
            )
            # self._messages.add_message(
            #     agent=self._supervisor, role="assistant", message=reply
            # )
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
        print("IN NEXT STEP", next_task)
        if next_task:
            self.set_current_task_agent(next_task.agent)
            self.set_current_agent(next_task.agent)
            next_message = next_task.prompt
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
            generating_agent=self.name,
            is_termination_message=is_termination_message,
        )

    async def _generate_executed_reply(
        self, messages: list[AgentMessage], sender: Agent
    ) -> AgentMessage:
        """Generate a reply after execution.

        If the execution does not error, return the last executor response which could do some
        parsing of the output.

        Otherwise, return the first error message.
        """
        executor_response = None
        default_response = AgentMessage(
            role="assistant",
            content=messages[-1].content,
            display_content=messages[-1].display_content,
            generating_agent=self.name,
        )
        last_message = messages[-1]
        print("IN EXECUTOR REPLY", self._agent_executors.get(sender, []))
        if messages[-1].requires_execution and self._agent_executors.get(sender, []):
            for executor in self._agent_executors.get(sender, []):
                # Generate a new chat between sender and executor
                sub_controller = ControllerAgent(
                    supervisor=sender,
                    # Very simply planner that always sends instruction to executor
                    # in a single step.
                    planner=PlannerAgent(
                        available_agents=[executor],
                        client=None,
                        llm_config=None,
                        database=None,
                    ),
                    supervisor_auto_respond=True,
                    silent=self._silent,
                )
                print("SENDER TO CHANGE TO SUPERVISOR", sender)
                sender.set_chat_role(AgentRole.SUPERVISOR)
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
                print("SENDER TO CHANGE TO EXEC", sender)
                sender.set_chat_role(AgentRole.EXECUTOR)
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
        print("^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^")
        print("SUPER", self._supervisor, "AGENT", self._current_agent)
        print("^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^")
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
                    generating_agent=self.name,
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
            role="user", content=input, generating_agent=self._supervisor.name
        )
        await self.receive(message, self._supervisor)
        all_messages = self._messages.get_messages_linearly_by_time()
        # Find last message that isn't with end tags
        # It's okay if the last message is a termination message, it can still have
        # content. We want to avoid the end tags.
        for msg in all_messages:
            print("MSG SUP", msg.content, "\n\n----\n\n", msg.display_content)
            print()
        for message in reversed(all_messages):
            if not Commands.has_end(message.content) and not Commands.has_next(
                message.content
            ):
                return message
        return None
