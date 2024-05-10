from meadow.agent.agent import Agent
from meadow.agent.schema import AgentMessage
from meadow.agent.user import UserAgent
from meadow.agent.utils import has_signal_string, print_message


class EvalUserAgent(UserAgent):
    """User agent for evaluation."""

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
        reply, to_send = await self.generate_reply(messages=[message], sender=sender)
        if reply.is_termination_message:
            return
        await self.send(reply, to_send)

    async def generate_reply(
        self,
        messages: list[AgentMessage],
        sender: Agent,
    ) -> tuple[AgentMessage, "Agent"]:
        """Generate a reply."""
        if has_signal_string(messages[-1].content, "<exit>"):
            return AgentMessage(
                role="assistant",
                content="Goodbye!",
                generating_agent=self.name,
                is_termination_message=True,
            ), sender
        return AgentMessage(
            role="assistant",
            content="<next>",
            generating_agent=self.name,
        ), sender
