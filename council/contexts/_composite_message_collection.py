from typing import Iterable, List

from ._chat_message import ChatMessage
from ._message_collection import MessageCollection


class CompositeMessageCollection(MessageCollection):
    """
    Wraps multiple :class:`MessageCollection` as one.
    """

    def __init__(self, collections: List[MessageCollection]) -> None:
        self._collections: List[MessageCollection] = collections

    @property
    def messages(self) -> Iterable[ChatMessage]:
        for collection in self._collections:
            yield from collection.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        for collection in reversed(self._collections):
            yield from collection.reversed
