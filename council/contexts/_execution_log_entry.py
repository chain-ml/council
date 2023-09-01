from datetime import datetime
from typing import List, Sequence

from .messages import ChatMessage
from .budget import Consumption


class ExecutionLogEntry:
    def __init__(self, source: str):
        self._source = source
        self._start = datetime.utcnow()
        self._duration = 0
        self._error = None
        self._consumptions: List[Consumption] = []
        self._messages: List[ChatMessage] = []

    @property
    def source(self) -> str:
        return self._source

    def log_consumption(self, consumption: Consumption):
        self._consumptions.append(consumption)

    def log_consumptions(self, consumptions: Sequence[Consumption]):
        [self.log_consumption(item) for item in consumptions]

    def log_message(self, message: ChatMessage):
        self._messages.append(message)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._duration = (datetime.utcnow() - self._start).total_seconds()
        self._error = exc_val

    def __repr__(self):
        return f"ExecutionLogEntry(source={self._source}, start={self._start}, duration={self._duration}, error={self._error})"
