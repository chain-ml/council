import json
from typing import Any, Dict

from ._execution_log_entry import ExecutionLogEntry


class ExecutionLog:
    def __init__(self):
        self._entries = []

    def new_entry(self, name: str) -> ExecutionLogEntry:
        result = ExecutionLogEntry(name)
        self._entries.append(result)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        result = {"entries": [item.to_dict() for item in self._entries]}

        return result
