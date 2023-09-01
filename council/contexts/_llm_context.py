from council.monitors import Monitored

from ._agent_context_store import AgentContextStore
from ._chat_history import ChatHistory
from ._context_base import ContextBase
from ._execution_context import ExecutionContext


class LLMContext(ContextBase):
    _execution_context: ExecutionContext

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext):
        super().__init__(store, execution_context)

    @staticmethod
    def from_context(context: ContextBase, monitored: Monitored) -> "LLMContext":
        return LLMContext(context._store, context._execution_context.new_for(monitored))

    @staticmethod
    def new_fake():
        return LLMContext(AgentContextStore(ChatHistory()), ExecutionContext())

    def new_for(self, monitored: Monitored) -> "LLMContext":
        return self.from_context(self, monitored)
