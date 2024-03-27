"""

Module: `agent_result`

This module provides the `AgentResult` class, which is used to encapsulate the result of an agent's operation by tracking scored chat messages.

Classes:
    AgentResult: A class to store and manage the results produced by an agent in the form of scored chat messages.

Attributes:
    _messages (List[ScoredChatMessage]): A private list that stores instances of `ScoredChatMessage`.

Properties:
    messages (Sequence[ScoredChatMessage]): Public read-only property that returns the list of scored chat messages.
    best_message (ChatMessage): Public read-only property that retrieves the message with the highest score from the list of scored messages.
    try_best_message (Option[ChatMessage]): Public read-only property that tries to retrieve the best message and wraps it in an `Option` type, which represents the presence or absence of a value.

Methods:
    __init__(self, messages: Optional[List[ScoredChatMessage]]=None):
        Initializes a new instance of the `AgentResult` class.
        Args:
            messages (Optional[List[ScoredChatMessage]]): A list of `ScoredChatMessage` instances to initialize the `AgentResult` with.
            If `None`, an empty list will be used.

Note:
    This module also imports several types from the `collections.abc` and `typing` modules, as well as specific classes from the `council.contexts` and `council.utils` modules.


"""
from collections.abc import Sequence
from typing import List, Optional

from council.contexts import ScoredChatMessage, ChatMessage
from council.utils import Option


class AgentResult:
    """
    Class representing the result of an agent's operation, encapsulating scored chat messages.
    This class stores messages which have a score associated with them, and provides
    methods to access these messages and retrieve the one with the best score.
    
    Attributes:
        _messages (List[ScoredChatMessage]):
             A private list of scored chat messages.
    
    Args:
        messages (Optional[List[ScoredChatMessage]]):
             An optional list of scored chat messages to initialize the
            result. If not provided, the default is an empty list.
        Properties:
        messages (Sequence[ScoredChatMessage]):
             Get the sequence of scored chat messages.
        best_message (ChatMessage):
             Get the chat message with the highest score.
        try_best_message (Option[ChatMessage]):
             Attempt to get the chat message with the highest score, wrapped in
            an Option type that handles the case where there are no messages.
    
    Raises:
        ValueError:
             If the operation to determine the best message is performed when there are no messages.

    """

    _messages: List[ScoredChatMessage]

    def __init__(self, messages: Optional[List[ScoredChatMessage]] = None):
        """
        Initializes a new instance of the class with an optional list of messages.
        This constructor method sets the instance's '_messages' attribute to the passed list of messages if provided,
        otherwise, it initializes it to an empty list. This is useful when an initial set of scored chat messages is available to be stored in the instance at creation time.
        
        Args:
            messages (Optional[List[ScoredChatMessage]]):
                 A list of ScoredChatMessage objects to initialize the '_messages' attribute. If `None`, the attribute is initialized to an empty list.
            

        """
        self._messages = messages if messages is not None else []

    @property
    def messages(self) -> Sequence[ScoredChatMessage]:
        """
        Gets the sequence of ScoredChatMessage objects associated with the object.
        This property returns a sequence of ScoredChatMessage objects which represent the
        messages that have been scored in a particular context. Accessing this property does not change
        the internal state of the object. The messages are returned in the form of a sequence, which can
        be an iterable item like a list or tuple, depending on the implementation.
        
        Returns:
            (Sequence[ScoredChatMessage]):
                 A sequence of ScoredChatMessage objects representing
                the scored chat messages.

        """
        return self._messages

    @property
    def best_message(self) -> ChatMessage:
        """
        Property that returns the best `ChatMessage` instance based on the highest score.
        This property assesses all messages within the `_messages` collection, identifying the one with the highest
        `score` attribute, and returns the `message` attribute of that `ChatMessage` instance. If there are multiple
        messages with the same highest score, it returns the `message` of the first one encountered in `_messages`.
        
        Returns:
            (ChatMessage):
                 The message with the highest score among all the stored messages.
        
        Raises:
            ValueError:
                 If the `_messages` collection is empty, meaning there is no message to evaluate.
            

        """
        return max(self._messages, key=lambda item: item.score).message

    @property
    def try_best_message(self) -> Option[ChatMessage]:
        """
        Property that returns the best chat message available.
        If there are no messages in the list, it returns an 'Option.none()'. If there are messages,
        it returns 'Option.some()' with the best message determined by the implemented criteria.
        
        Returns:
            (Option[ChatMessage]):
                 An 'Option' container which either carries the best message
                ('Option.some(ChatMessage)') or signifies no messages are available
                ('Option.none()').
            

        """
        if len(self._messages) == 0:
            return Option.none()
        return Option.some(self.best_message)
