from typing import Iterable, Optional, Sequence

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._message_collection import MessageCollection
from ._monitored import Monitored


class AgentContext(ContextBase):
    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        super().__init__(store, execution_context, budget)

    @staticmethod
    def empty(budget: Optional[Budget] = None) -> "AgentContext":
        return AgentContext.from_chat_history(ChatHistory(), budget)

    @staticmethod
    def from_chat_history(chat_history: ChatHistory, budget: Optional[Budget] = None) -> "AgentContext":
        store = AgentContextStore(chat_history)
        return AgentContext(store, ExecutionContext(store.execution_log, "agent"), budget or Budget.default())

    @staticmethod
    def from_user_message(message: str, budget: Optional[Budget] = None) -> "AgentContext":
        return AgentContext.from_chat_history(ChatHistory.from_user_message(message), budget)

    def new_agent_context_for(self, monitored: Monitored) -> "AgentContext":
        return AgentContext(self._store, self._execution_context.new_for(monitored), self._budget)

    def new_iteration(self):
        self._store.new_iteration()

    def new_agent_context_for_new_iteration(self) -> "AgentContext":
        self.new_iteration()
        name = f"iterations[{len(self._store.iterations) - 1}]"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    def new_agent_context_for_execution_unit(self, name: str) -> "AgentContext":
        name = f"execution({name})"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    @property
    def chains(self) -> Iterable[MessageCollection]:
        return self._store.current_iteration.chains.values()

    @property
    def evaluation(self) -> Sequence[ScoredChatMessage]:
        return self._store.current_iteration.evaluator

    def set_evaluation(self, messages: Iterable[ScoredChatMessage]):
        self._store.current_iteration.set_evaluator(messages)
