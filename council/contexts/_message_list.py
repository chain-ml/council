from typing import Any, Iterable, List, Optional

from ._chat_message import ChatMessage
from ._message_collection import MessageCollection


class MessageList(MessageCollection):
    """
    represents an appendable list of :class:`.ChatMessage`
    """

    def __init__(self, messages: Optional[Iterable[ChatMessage]] = None) -> None:
        """
        initialize a new instance
        """

        self._messages: List[ChatMessage] = []
        if messages is not None:
            self._messages.extend(messages)

    @property
    def messages(self) -> Iterable[ChatMessage]:
        return self._messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        return reversed(self._messages)

    def add_user_message(self, message: str, data: Optional[Any] = None) -> None:
        """
        adds a user :class:`ChatMessage` into the history

        Arguments:
            message (str): a text message
            data (Any): Optional data to attach to the message
        """

        self._messages.append(ChatMessage.user(message, data))

    def add_agent_message(self, message: str, data: Any = None) -> None:
        """
        adds an agent class:`ChatMessage` into the history

        Arguments:
            message (str): a text message
            data (Any): Optional data to attach to the message
        """

        self._messages.append(ChatMessage.agent(message, data))

    def add_message(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def add_messages(self, messages: Iterable[ChatMessage]) -> None:
        self._messages.extend(messages)

    def __len__(self):
        return len(self._messages)
