from typing import Iterable, List

from ._chat_message import ChatMessage
from ._message_collection import MessageCollection


class CompositeMessageCollection(MessageCollection):
    """
    Wraps multiple :class:`MessageCollection` as one.
    """

    _collections: List[MessageCollection]

    def __init__(self, collections: List[MessageCollection]):
        self._collections = collections

    @property
    def messages(self) -> Iterable[ChatMessage]:
        for collection in self._collections:
            yield from collection.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        for collection in reversed(self._collections):
            yield from collection.reversed
