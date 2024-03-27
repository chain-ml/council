"""

A module that wraps the MessageList to additionally log messages as they are appended to the list.

This module provides a MonitoredMessageList class which inherits from the MessageCollection abstract base class, adding the capability to log each message using an ExecutionLogEntry instance before the message is added to the internal list.

Classes:
    MonitoredMessageList: A collection of chat messages that also records each message via an ExecutionLogEntry.



"""
from typing import Iterable

from ._chat_message import ChatMessage
from ._execution_log_entry import ExecutionLogEntry
from ._message_collection import MessageCollection
from ._message_list import MessageList


class MonitoredMessageList(MessageCollection):
    """
    A list-like class responsible for storing and logging chat messages within a MonitoredMessageList object.
    This class acts as a wrapper around another MessageCollection object, allowing for additional
    functionality such as the ability to log each message that is appended to the list.
    
    Attributes:
        _inner (MessageList):
             An instance of MessageList which is being monitored.
        Properties:
        messages (Iterable[ChatMessage]):
             Provides an iterable of ChatMessage objects contained in the inner MessageList.
        reversed (Iterable[ChatMessage]):
             Provides an iterable of ChatMessage objects in reverse order contained in the inner MessageList.
    
    Methods:
        __init__(self, message_list:
             MessageList): Initializes a new instance of MonitoredMessageList.
        append(self, message:
             ChatMessage, log_entry: ExecutionLogEntry): Appends a ChatMessage to the monitored MessageList and logs the operation.

    """
    def __init__(self, message_list: MessageList):
        """
        Initializes an instance of the enclosing class with a given MessageList object.
        
        Args:
            message_list (MessageList):
                 An instance of MessageList to be encapsulated by the enclosing object.
            Description:
                This constructor method is called when a new instance of the containing class is created. It initializes the object’s internal state by setting an `_inner` attribute to the value of the `message_list` passed to it. This allows the object to maintain a reference to the given MessageList object, presuming that MessageList is a defined class or type in the context which supports certain operations expected by the containing class.

        """
        self._inner = message_list

    @property
    def messages(self) -> Iterable[ChatMessage]:
        """
        Gets the messages from the current chat instance.
        This property returns an iterable of ChatMessage objects, which represent
        the messages contained within the chat instance it is called upon.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable collection of ChatMessage instances.

        """
        return self._inner.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        """
        
        Returns an iterable containing the chat messages in reversed order.
            The property provides an iterable over the internal `_inner` collection's `.reversed` attribute, allowing the consumer to iterate through chat messages starting from the most recent one to the oldest.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable object that allows iterating through chat messages in reverse chronological order.

        """
        return self._inner.reversed

    def append(self, message: ChatMessage, log_entry: ExecutionLogEntry):
        """
        Appends a message to an execution log and to an internal message container.
        This function takes a ChatMessage and an ExecutionLogEntry as parameters. First, the message is logged to the ExecutionLogEntry. Then, the message is added to the internal message container of the current instance.
        
        Args:
            message (ChatMessage):
                 The message object to be added. Must be an instance of the ChatMessage class.
            log_entry (ExecutionLogEntry):
                 The execution log entry to which the message should be logged.
        
        Raises:
            TypeError:
                 If the 'message' is not an instance of ChatMessage or if 'log_entry' is not an instance of ExecutionLogEntry.

        """
        log_entry.log_message(message)
        self._inner.add_message(message)
