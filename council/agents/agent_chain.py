from typing import Optional, Any

from .agent import Agent
from council.chains import Chain
from council.contexts import AgentContext, ChainContext, ChatMessageBase
from council.runners import Budget, RunnerExecutor


class AgentChain(Chain):
    def __init__(self, name: str, description: str, agent: Agent):
        super().__init__(name, description, [])
        self.agent = agent

    def execute(
        self,
        context: ChainContext,
        budget: Budget,
        executor: Optional[RunnerExecutor] = None,
    ) -> Any:
        agent_context = AgentContext(context.chatHistory)
        result = self.agent.execute(agent_context, budget)
        maybe_message = result.try_best_message
        if maybe_message.is_some():
            message = maybe_message.unwrap()
            context.current.append(
                ChatMessageBase.skill(message.message, message.data, message.source, message.is_error)
            )
