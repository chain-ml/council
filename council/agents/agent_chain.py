from typing import Any, Optional

from council.chains import Chain
from council.contexts import AgentContext, ChainContext, ChatMessage, Monitored
from council.runners import RunnerExecutor
from .agent import Agent


class AgentChain(Chain):
    _agent: Monitored[Agent]

    def __init__(self, name: str, description: str, agent: Agent):
        super().__init__(name, description, [])
        self._agent = self.new_monitor("agent", agent)

    @property
    def agent(self) -> Agent:
        return self._agent.inner

    def _execute(
        self,
        context: ChainContext,
        executor: Optional[RunnerExecutor] = None,
    ) -> Any:
        result = self.agent.execute(AgentContext.from_chat_history(context.chat_history))
        maybe_message = result.try_best_message
        if maybe_message.is_some():
            message = maybe_message.unwrap()
            context.append(ChatMessage.skill(message.message, message.data, message.source, message.is_error))
