from typing import Optional

from council.monitors import Monitored
from ._budget import Budget, InfiniteBudget

from ._agent_context_store import AgentContextStore
from ._chat_history import ChatHistory
from ._context_base import ContextBase
from ._execution_context import ExecutionContext


class ScorerContext(ContextBase):
    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        super().__init__(store, execution_context, budget)

    @staticmethod
    def from_context(context: ContextBase, monitored: Monitored, budget: Optional[Budget] = None) -> "ScorerContext":
        return ScorerContext(context._store, context._execution_context.new_for(monitored), budget or context._budget)

    @staticmethod
    def empty() -> "ScorerContext":
        return ScorerContext(AgentContextStore(ChatHistory()), ExecutionContext(), InfiniteBudget())

    def new_for(self, monitored: Monitored) -> "ScorerContext":
        return self.from_context(self, monitored)
