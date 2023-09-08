from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._execution_context import ExecutionContext
from ._execution_log_entry import ExecutionLogEntry
from ._monitored import Monitored
from ._monitored_budget import MonitoredBudget


class ContextBase:
    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        self._store = store
        self._execution_context = execution_context
        self._budget = MonitoredBudget(execution_context.entry, budget)

    @property
    def iteration_count(self) -> int:
        return len(self._store.iterations)

    @property
    def log_entry(self) -> ExecutionLogEntry:
        return self._execution_context.entry

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def chat_history(self) -> ChatHistory:
        return self._store.chat_history

    def __enter__(self):
        self.log_entry.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_entry.__exit__(exc_type, exc_val, exc_tb)

    def new_log_entry(self, monitored: Monitored) -> ExecutionLogEntry:
        return self._execution_context.new_for(monitored).entry
