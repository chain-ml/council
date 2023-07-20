from typing import List, Optional, Sequence

from council.contexts import ChatMessage


class RunnerResult:
    def __init__(self, messages: Optional[List[ChatMessage]] = None, error: Optional[Exception] = None):
        self._messages = messages or []
        self._error = error

    @property
    def messages(self) -> Sequence[ChatMessage]:
        return self._messages

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def is_error(self) -> bool:
        return self._error is not None

    @staticmethod
    def empty() -> "RunnerResult":
        return RunnerResult(messages=[])

    def add(self, other: "RunnerResult") -> None:
        self._messages.extend(other.messages)
        if self._error is None:
            self._error = other.error
