from typing import List

from council.core import ControllerBase, AgentContext, Chain, Budget
from council.core.execution_context import ScoredAgentMessage
from council.core.execution_unit import ExecutionUnit


class BasicController(ControllerBase):
    """a basic controller that requests all chains to be executed and returns all results"""

    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        return [ExecutionUnit(chain, budget) for chain in chains]

    def select_responses(self, context: AgentContext) -> List[ScoredAgentMessage]:
        return context.evaluationHistory[-1]
