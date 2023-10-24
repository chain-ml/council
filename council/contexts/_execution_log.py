import json
from typing import Any, Dict, Optional

from ._monitorable import Monitorable
from ._execution_log_entry import ExecutionLogEntry


class ExecutionLog:
    """
    represents the log of execution for each executable items (i.e. :class:`~council.agents.Agent`,
    :class:`~council.chains.Chain`, :class:`~council.skills.SkillBase` ...)
    """

    def __init__(self):
        self._entries = []

    def new_entry(self, name: str, node: Optional[Monitorable]) -> ExecutionLogEntry:
        """
        adds a new entry into the log
        Args:
            name: name of the new entry
            node: the related monitored runner

        Returns:
            the newly added entry
        """
        result = ExecutionLogEntry(name, node)
        self._entries.append(result)
        return result

    def to_json(self) -> str:
        """
        serialize the execution log as a `json` string

        Returns:
            a `json` string
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        convert into a dictionary
        """
        result = {"entries": [item.to_dict() for item in self._entries]}

        return result
