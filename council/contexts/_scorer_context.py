from typing import Optional

from ._agent_context_store import AgentContextStore
from ._budget import Budget, InfiniteBudget
from ._chat_history import ChatHistory
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._monitored import Monitored


class ScorerContext(ContextBase):
    """
    represents a context used by a :class:`~council.scorer.ScorerBase`
    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        super().__init__(store, execution_context, budget)

    @staticmethod
    def from_context(context: ContextBase, monitored: Monitored, budget: Optional[Budget] = None) -> "ScorerContext":
        """
        creates a new instance from the given context, adjusting the execution appropriately
        """
        return ScorerContext(context._store, context._execution_context.new_for(monitored), budget or context._budget)

    @staticmethod
    def empty() -> "ScorerContext":
        """
        helper function that creates a new empty instance

        For test purpose only.
        """
        return ScorerContext(AgentContextStore(ChatHistory()), ExecutionContext(), InfiniteBudget())

    def new_for(self, monitored: Monitored) -> "ScorerContext":
        """
        returns a new instance for the given object, adjusting the execution context appropriately
        """
        return self.from_context(self, monitored)
