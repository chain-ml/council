from typing import List

from council.contexts import (
    AgentContext,
    ScoredChatMessage,
    ChatMessage,
)
from council.runners.budget import Budget
from .evaluator_base import EvaluatorBase


class BasicEvaluator(EvaluatorBase):
    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        result = []
        for chain_messages in context.chains:
            chain_result = chain_messages.last_message
            if chain_result is None:
                continue
            score = 1 if chain_result.is_kind_skill and chain_result.is_ok else 0
            result.append(
                ScoredChatMessage(
                    ChatMessage.agent(chain_result.message, chain_result.data, is_error=chain_result.is_error),
                    score,
                )
            )
        return result
