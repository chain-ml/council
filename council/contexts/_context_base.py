from typing import Any, Dict

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._context_logger import ContextLogger
from ._execution_context import ExecutionContext
from ._execution_log_entry import ExecutionLogEntry
from ._monitored import Monitored
from ._monitored_budget import MonitoredBudget


class ContextBase:
    """
    base class for context.

    It provides a secured and monitored access to the data generated during the execution.
    The actual data are stored in the underlying `class`:AgentContextStore`.
    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        self._store = store
        self._execution_context = execution_context
        self._budget = MonitoredBudget(execution_context.entry, budget)
        self._logger = ContextLogger(execution_context.entry)

    @property
    def iteration_count(self) -> int:
        """
        the number of iteration for this execution
        """
        return len(self._store.iterations)

    @property
    def log_entry(self) -> ExecutionLogEntry:
        """
        the log entry
        """
        return self._execution_context.entry

    @property
    def budget(self) -> Budget:
        """
        the budget
        """
        return self._budget

    @property
    def chat_history(self) -> ChatHistory:
        """
        the chat history
        """
        return self._store.chat_history

    @property
    def logger(self) -> ContextLogger:
        return self._logger

    def __enter__(self):
        self.log_entry.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_entry.__exit__(exc_type, exc_val, exc_tb)

    def new_log_entry(self, monitored: Monitored) -> ExecutionLogEntry:
        """
        creates a new log entry from this context
        """
        return self._execution_context.new_for(monitored).entry

    def execution_log_to_dict(self) -> Dict[str, Any]:
        """
        returns the execution log as a dictionary
        """
        return self._execution_context.execution_log.to_dict()

    def execution_log_to_json(self) -> str:
        """
        returns the execution as a JSON string
        """
        return self._execution_context.execution_log.to_json()
