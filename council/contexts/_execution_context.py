from __future__ import annotations

from typing import Optional

from ._execution_log import ExecutionLog
from ._execution_log_entry import ExecutionLogEntry
from ._monitorable import Monitorable
from ._monitored import Monitored


class ExecutionContext:
    """
    context storing the execution information
    """

    def __init__(
        self, execution_log: Optional[ExecutionLog] = None, path: str = "", node: Optional[Monitorable] = None
    ) -> None:
        self._executionLog: ExecutionLog = execution_log or ExecutionLog()
        self._entry: ExecutionLogEntry = self._executionLog.new_entry(path, node)

    def _new_path(self, name: str) -> str:
        return name if self._entry.source == "" else f"{self._entry.source}/{name}"

    def new_from_name(self, name: str) -> ExecutionContext:
        """
        returns a new instance for the given name
        """
        return ExecutionContext(self._executionLog, self._new_path(name))

    def new_for(self, monitored: Monitored) -> ExecutionContext:
        """
        returns a new instance for the given object
        """
        return ExecutionContext(self._executionLog, self._new_path(monitored.name), monitored.inner)

    @property
    def entry(self) -> ExecutionLogEntry:
        """
        the current log entry
        """
        return self._entry

    @property
    def execution_log(self) -> ExecutionLog:
        """
        the execution log
        """
        return self._executionLog
