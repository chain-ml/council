from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._monitorable import Monitorable
from ._budget import Consumption
from ._chat_message import ChatMessage


class ExecutionLogEntry:
    """
    represents one entry in the :class:`ExecutionLog`
    """

    def __init__(self, source: str, node: Optional[Monitorable]):
        self._source = source
        self._node = node
        self._start = datetime.now(timezone.utc)
        self._duration = 0
        self._error = None
        self._consumptions: List[Consumption] = []
        self._messages: List[ChatMessage] = []
        self._logs: List[Tuple[datetime, str, str]] = []

    @property
    def source(self) -> str:
        """
        the source/name of the entry
        """
        return self._source

    @property
    def node(self) -> Optional[Monitorable]:
        """
        the related monitorable node in the execution graph
        """

        return self._node

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
            "logs": self._logs_to_dict(),
        }

        if self._error is not None:
            result["error"] = self._error

        if self._node is not None:
            result["node"] = self._node.render_as_dict(include_children=False)

        return result

    def _log_message(self, level: str, message: str, *args: Any) -> None:
        msg = message % args if len(args) > 0 else message
        self._logs.append((datetime.now(timezone.utc), level, msg))

    def log_debug(self, message: str, *args: Any) -> None:
        self._log_message("DEBUG", message, *args)

    def log_info(self, message: str, *args: Any) -> None:
        self._log_message("INFO", message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        self._log_message("WARNING", message, *args)

    def log_error(self, message: str, *args: Any) -> None:
        self._log_message("ERROR", message, *args)

    def _logs_to_dict(self) -> List[Dict[str, Any]]:
        return [{"time": item[0].isoformat(), "level": item[1], "message": item[2]} for item in self._logs]
