from typing import Any, Dict

import logging

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
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

    def log_debug(self, message: str) -> None:
        """
        Logs a debug message using `logging` and keeps track of it into the context

        Args:
            message (str): a message
        """
        if self._logger_log(logging.DEBUG, message):
            self.log_entry.log_debug(message)

    def log_error(self, message: str) -> None:
        """
        Logs an error message using `logging` and keeps track of it into the context

        Args:
            message (str): a message
        """
        if self._logger_log(logging.ERROR, message):
            self.log_entry.log_error(message)

    def log_info(self, message: str) -> None:
        """
        Logs an info message using `logging` and keeps track of it into the context

        Args:
            message (str): a message
        """
        if self._logger_log(logging.INFO, message):
            self.log_entry.log_info(message)

    @staticmethod
    def _logger_log(level: int, message: str) -> bool:
        import inspect

        stack = inspect.stack()
        logger_name = stack[2].frame.f_globals["__name__"]
        logger = logging.getLogger(logger_name)
        logger.log(level, message, stacklevel=3)
        return logger.isEnabledFor(level)
