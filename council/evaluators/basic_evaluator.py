from typing import List

from council.contexts import (
    AgentContext,
    ScoredChatMessage,
    ChatMessage,
)
from council.runners.budget import Budget
from .evaluator_base import EvaluatorBase


class BasicEvaluator(EvaluatorBase):
    """
    A BasicEvaluator that carries along the last Skill source of each Chain ExecutionUnit.
    """

    def _execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        result: List[ScoredChatMessage] = []
        for chain_history in context.chainHistory.values():
            chain_result = chain_history[-1].messages[-1]
            score = 1.0 if chain_result.is_kind_skill and chain_result.is_ok else 0.0
            result.append(
                ScoredChatMessage(
                    ChatMessage.agent(
                        message=chain_result.message,
                        data=chain_result.data,
                        source=chain_result.source,
                        is_error=chain_result.is_error,
                    ),
                    score,
                )
            )
        return result
