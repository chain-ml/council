from typing import List

from council.contexts import (
    AgentContext,
    ScoredChatMessage,
    ChatMessage,
)
from .evaluator_base import EvaluatorBase


class BasicEvaluator(EvaluatorBase):
    """
    A BasicEvaluator that carries along the last Skill source of each Chain ExecutionUnit.
    """

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        result: List[ScoredChatMessage] = []
        for chain_messages in context.chains:
            chain_result = chain_messages.last_message
            if chain_result is None:
                continue
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
