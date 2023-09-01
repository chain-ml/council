from ._execution_log_entry import ExecutionLogEntry


class ExecutionLog:
    def __init__(self):
        self._entries = []

    def new_entry(self, name: str) -> ExecutionLogEntry:
        result = ExecutionLogEntry(name)
        self._entries.append(result)
        return result
