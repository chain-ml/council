from typing import Iterable

from ._chat_message import ChatMessage
from ._execution_log_entry import ExecutionLogEntry
from ._message_collection import MessageCollection
from ._message_list import MessageList


class MonitoredMessageList(MessageCollection):
    def __init__(self, message_list: MessageList):
        self._inner = message_list

    @property
    def messages(self) -> Iterable[ChatMessage]:
        return self._inner.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        return self._inner.reversed

    def append(self, message: ChatMessage, log_entry: ExecutionLogEntry):
        log_entry.log_message(message)
        self._inner.add_message(message)
