from typing import Optional

from council.chains import ChainBase
from council.contexts import AgentContext, ChainContext, ChatMessage, Monitored
from council.runners import RunnerExecutor

from .agent import Agent


class AgentChain(ChainBase):
    """
    A chain that wraps an Agent, so that it can be invoked from another agent.
    """

    def __init__(self, name: str, description: str, agent: Agent) -> None:
        """
        Initialize a new instance.

        Args:
            name (str): The name of the chain.
            description (str): The description of the chain.
            agent (Agent): The agent to be wrapped
        """
        super().__init__(name, description)
        self._agent: Monitored[Agent] = self.new_monitor("agent", agent)

    @property
    def agent(self) -> Agent:
        return self._agent.inner

    def _execute(self, context: ChainContext, _executor: Optional[RunnerExecutor] = None) -> None:
        result = self.agent.execute(AgentContext.from_chat_history(context.chat_history))
        maybe_message = result.try_best_message
        if maybe_message.is_some():
            message = maybe_message.unwrap()
            context.append(ChatMessage.skill(message.message, message.data, message.source, message.is_error))
