from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

from ._budget import Consumption
from ._chat_message import ChatMessage


class ExecutionLogEntry:
    """
    represents one entry in the :class:`ExecutionLog`
    """

    def __init__(self, source: str):
        self._source = source
        self._start = datetime.now(timezone.utc)
        self._duration = 0
        self._error = None
        self._consumptions: List[Consumption] = []
        self._messages: List[ChatMessage] = []

    @property
    def source(self) -> str:
        """
        the source/name of the entry
        """
        return self._source

    def log_consumption(self, consumption: Consumption) -> None:
        """
        logs a budget's :class:`Consumption`
        """
        self._consumptions.append(consumption)

    def log_consumptions(self, consumptions: Sequence[Consumption]) -> None:
        """
        logs multiple budget's :class:`Consumption`
        """
        for consumption in consumptions:
            self.log_consumption(consumption)

    def log_message(self, message: ChatMessage) -> None:
        """
        logs a :class:`ChatMessage`
        """
        self._messages.append(message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._duration = (datetime.now(timezone.utc) - self._start).total_seconds()
        self._error = exc_val

    def __repr__(self):
        return (
            "ExecutionLogEntry("
            f"source={self._source}, start={self._start}, duration={self._duration}, error={self._error}"
            ")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        convert into a dictionary
        """
        result = {
            "source": self._source,
            "start": self._start.isoformat(),
            "duration": self._duration,
            "consumptions": [item.to_dict() for item in self._consumptions],
            "messages": [item.to_dict() for item in self._messages],
        }

        if self._error is not None:
            result["error"] = self._error

        return result
