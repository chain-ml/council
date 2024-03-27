"""

Module: _composite_message_collection

This module defines a class for holding a collection of message collections, allowing
aggregated access to messages from multiple contained collections in a composite manner.

Classes:
    CompositeMessageCollection(MessageCollection):
        A collection that combines multiple message collections into a single composite entity.
        It provides a unified interface for iterating over messages from all contained collections,
        both in forward and reverse order.

        Attributes:
            _collections (List[MessageCollection]): A list of MessageCollection instances that
                are aggregated by this CompositeMessageCollection.

        Methods:
            __init__(self, collections: List[MessageCollection]): Initializes a new instance of
                CompositeMessageCollection with the provided list of MessageCollection instances.

            messages (Iterable[ChatMessage]): A property that yields all messages from the contained
                collections in the order they appear within each collection.

            reversed (Iterable[ChatMessage]): A property that yields all messages from the contained
                collections in reverse order. It starts with messages from the last collection and
                proceeds backwards through the collections.


"""
from typing import Iterable, List

from ._chat_message import ChatMessage
from ._message_collection import MessageCollection


class CompositeMessageCollection(MessageCollection):
    """
    A class that aggregates multiple MessageCollection instances into a single composite collection.
    This class allows for the composition of multiple message collections to be treated as one unified collection. It provides iterable access to all the messages across the component collections. Messages can be iterated in the order they are held within the internal collections or in reverse order.
    
    Attributes:
        _collections (List[MessageCollection]):
             A private list of MessageCollection instances that form the composite.
    
    Methods:
        __init__(collections:
             List[MessageCollection]): Initializes a new instance of CompositeMessageCollection with a list of MessageCollections.
        messages:
             Provides an iterable of ChatMessage objects from the aggregated message collections.
        reversed:
             Provides an iterable of ChatMessage objects from the aggregated message collections, but in reverse order.
            Each property is designed as a generator to yield the messages lazily, thus enhancing efficiency for large collections of messages.

    """

    _collections: List[MessageCollection]

    def __init__(self, collections: List[MessageCollection]):
        """
        Initializes a new instance of the class with a list of MessageCollection objects.
        
        Args:
            collections (List[MessageCollection]):
                 A list of MessageCollection instances to be associated with this object.

        """
        self._collections = collections

    @property
    def messages(self) -> Iterable[ChatMessage]:
        """
        Generator function that iterates over a series of message collections to retrieve individual messages.
        This function is implemented as a property, effectively turning it into a 'getter' for the messages from all collections
        combined. It loops through each collection in the instance's `_collections` attribute and yields messages one by
        one, flattening the messages from all different collections into a single iterable sequence.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterator over `ChatMessage` objects from all collections contained in `_collections`.
            

        """
        for collection in self._collections:
            yield from collection.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        """
        Provides an iterable that returns chat messages in reverse order from multiple collections.
        This property method creates an iterator that traverses through a series of
        collections in reverse, yielding each chat message from these collections
        also in reverse order. It assumes `self._collections` is an iterable containing
        collections that themselves have a `reversed` property providing an iterable
        for accessing items in reverse order.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable of `ChatMessage` objects in reverse order.
            

        """
        for collection in reversed(self._collections):
            yield from collection.reversed
