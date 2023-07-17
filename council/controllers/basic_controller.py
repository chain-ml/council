from typing import List

from council.chains import Chain
from council.contexts import AgentContext, ScoredChatMessage
from council.runners import Budget

from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit


class BasicController(ControllerBase):
    """a basic controller that requests all chains to be executed and returns all results"""

    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        return [ExecutionUnit(chain, budget) for chain in chains]

    def select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        return context.evaluationHistory[-1]
