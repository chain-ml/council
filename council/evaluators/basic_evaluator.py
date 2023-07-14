from typing import List

from council.contexts import AgentContext, ScoredAgentMessage, SkillSuccessMessage, AgentMessage
from council.runners.budget import Budget
from .evaluator_base import EvaluatorBase


class BasicEvaluator(EvaluatorBase):
    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredAgentMessage]:
        result = []
        for chain_history in context.chainHistory.values():
            chain_result = chain_history[-1].messages[-1]
            score = 1 if isinstance(chain_result, SkillSuccessMessage) else 0
            result.append(ScoredAgentMessage(AgentMessage(chain_result.message, chain_result.data), score))
        return result
