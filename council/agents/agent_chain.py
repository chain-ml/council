from typing import Optional, Any

from .agent import Agent
from council.chains import Chain
from council.contexts import AgentContext, ChainContext, ChatMessage
from council.runners import Budget, RunnerExecutor
from ..monitors import Monitored


class AgentChain(Chain):
    _agent: Monitored[Agent]

    def __init__(self, name: str, description: str, agent: Agent):
        super().__init__(name, description, [])
        self._agent = self.new_monitor("agent", agent)

    @property
    def agent(self) -> Agent:
        return self._agent.inner

    def execute(
        self,
        context: ChainContext,
        budget: Budget,
        executor: Optional[RunnerExecutor] = None,
    ) -> Any:
        agent_context = AgentContext.from_chat_history(context.chat_history)
        result = self.agent.execute(agent_context, budget)
        maybe_message = result.try_best_message
        if maybe_message.is_some():
            message = maybe_message.unwrap()
            context.append(ChatMessage.skill(message.message, message.data, message.source, message.is_error))
