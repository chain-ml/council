from typing import Optional, Any

from council.core import Chain, Agent, AgentContext, ChainContext, Budget
from council.core.execution_context import SkillSuccessMessage
from council.core.runners import RunnerExecutor


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
            context.current.messages.append(SkillSuccessMessage(self.name, message.message, message.data))
