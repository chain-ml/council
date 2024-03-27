"""

A module providing an abstract base class, `MessageCollection`, for handling collections of chat messages.

The `MessageCollection` class is designed to be a flexible foundation for implementing collections of `ChatMessage`
instances, which can represent various types of messages (e.g., from users, agents, skills, etc.) within a chat system.
It outlines core properties and methods that access messages based on defined criteria such as the latest message, the last
message from a user, or the last message from an agent.

Attributes:
    None

Methods:
    messages: An abstract property that should be implemented to return an iterable of ChatMessage instances.
    reversed: An abstract property that should provide the messages iterable in reverse order.

    last_message: Gets the last message added to the collection if present.
    try_last_message: Attempts to fetch the last message and wraps it in an Option.
    last_user_message: Retrieves the last user-sent message in the collection.
    try_last_user_message: Attempts to fetch the last user-sent message and wraps it in an Option.
    last_agent_message: Fetches the last agent-sent message in the collection.
    try_last_agent_message: Attempts to fetch the last agent-sent message and wraps it in an Option.
    last_message_from_skill: Provides the last message from a specific skill given the skill name.
    try_last_message_from_skill: Attempts to fetch the last message from a specific skill and wraps it in an Option.
    _last_message_filter: A helper function to apply a filter predicate to find a specific last message.
    message_kind_predicate: Creates a predicate function for filtering messages of a specific ChatMessageKind.

This module also imports necessary dependencies, such as `ChatMessage` and `ChatMessageKind` classes,
types like `Iterable`, `Optional`, and `Callable`, and utility classes like `Option` from the module's package.


"""
import abc
from typing import Callable, Iterable, Optional

from more_itertools import first
from typing_extensions import TypeGuard

from ._chat_message import ChatMessage, ChatMessageKind
from ..utils import Option


class MessageCollection(abc.ABC):
    """
    A base class that defines the structure for a collection of chat messages.
    This abstract class provides the properties and methods required for managing a collection of ChatMessage instances. It
    enforces the implementation of certain properties that represent different views of the messages in the collection.
    Properties:
    messages (Iterable[ChatMessage]): An abstract property that must be overridden to return an iterable of ChatMessage objects.
    reversed (Iterable[ChatMessage]): An abstract property that must be overridden to return an iterable of ChatMessage objects in reversed order.
    
    Methods:
        last_message_from_skill(skill_name:
             str) -> Optional[ChatMessage]: Gets the last message from a specific skill.
        try_last_message_from_skill(skill_name:
             str) -> Option[ChatMessage]: Wraps the result of last_message_from_skill in an Option.
        _last_message_filter(predicate:
             Callable[[ChatMessage], bool]) -> Optional[ChatMessage]: Applies a predicate to filter messages and return the last one.
        message_kind_predicate(kind:
             ChatMessageKind) -> Callable[[ChatMessage], bool]: Returns a predicate function to filter messages by kind.
        The class also implements four concrete properties that utilize the abstract properties to provide specific information:
        - last_message:
             Returns the last message in the collection.
        - try_last_message:
             Optionally wraps the last message in an Option.
        - last_user_message:
             Returns the last message in the collection sent by a user.
        - try_last_user_message:
             Optionally wraps the last user message in an Option.
        - last_agent_message:
             Returns the last message in the collection sent by an agent.
        - try_last_agent_message:
             Optionally wraps the last agent message in an Option.
            It is assumed that the Option type has some semantics for the optional existence of a value, similar to Optional but
            possibly with additional functionality or semantics.

    """

    @property
    @abc.abstractmethod
    def messages(self) -> Iterable[ChatMessage]:
        """
        Fetches an iterable of chat messages associated with the instance. This method is an abstract property that should be implemented by subclasses to return the relevant iterable of ChatMessage objects. The property is read-only. Returns: Iterable[ChatMessage]: An iterable that allows traversing through the chat messages.

        """
        pass

    @property
    @abc.abstractmethod
    def reversed(self) -> Iterable[ChatMessage]:
        """
        
        Returns an iterable of chat messages in reverse order.
            This abstract method must be implemented by subclasses to provide the functionality
            of retrieving chat messages in the reverse order to how they were received or stored.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable object that on iteration yields chat messages
                in reverse order.

        """
        pass

    @property
    def last_message(self) -> Optional[ChatMessage]:
        """
        
        Returns the last message from a sequence of chat messages.
            This property retrieves the most recent chat message from an iterable sequence
            of `ChatMessage` objects. It utilizes the `first` function on the reversed order of
            messages, effectively returning the last item, or `None` if the sequence is empty.
        
        Returns:
            (Optional[ChatMessage]):
                 The most recent `ChatMessage` object or `None` if there
                are no messages in the sequence.

        """
        return first(self.reversed, None)

    @property
    def try_last_message(self) -> Option[ChatMessage]:
        """
        Attempts to retrieve the last chat message as an Option object.
        This property checks for the existence of the last chat message, wraps it in
        an Option instance, and returns it. If there is no last chat message,
        it returns an Option object representing the absence of a value (None).
        
        Returns:
            (Option[ChatMessage]):
                 An Option object containing the last chat message if it exists;
                otherwise, an Option object containing None.
            

        """
        return Option(self.last_message)

    @property
    def last_user_message(self) -> Optional[ChatMessage]:
        """
        
        Returns the last user message encountered in a chat.
            This property filters through received messages and retrieves the last message
            that qualifies as a user message, which it determines using a provided message kind
            predicate. If no such message exists, the result is `None`.
            To use this property, you should define a predicate function that accepts message kinds
            and returns a boolean indicating whether a particular message kind represents a user message.
        
        Returns:
            (Optional[ChatMessage]):
                 The last message that matches the user message kind as determined by the
                provided predicate, or `None` if no such message can be found.

        """
        return self._last_message_filter(self.message_kind_predicate(ChatMessageKind.User))

    @property
    def try_last_user_message(self) -> Option[ChatMessage]:
        """
        Attempts to retrieve the last message from a user within a chat session.
        This property will return an `Option` object encapsulating the last `ChatMessage`
        sent by the user. If there is no message, it will return an `Option` object containing `None`.
        
        Returns:
            (Option[ChatMessage]):
                 An `Option` object containing the last user `ChatMessage` if it exists,
                otherwise an `Option` containing `None`.
            

        """
        return Option(self.last_user_message)

    @property
    def last_agent_message(self) -> Optional[ChatMessage]:
        """
        Property that retrieves the last message from an agent within a chat session.
        This property accesses the internal chat message filtering method to obtain
        the most recent message that satisfies the `message_kind_predicate` for the
        `ChatMessageKind.Agent`. This message represents the last action or
        communication performed by an agent in the context of the current object.
        
        Returns:
            (Optional[ChatMessage]):
                 The last `ChatMessage` instance sent by an agent,
                or None if no such message exists.
            

        """
        return self._last_message_filter(self.message_kind_predicate(ChatMessageKind.Agent))

    @property
    def try_last_agent_message(self) -> Option[ChatMessage]:
        """
        Gets an Option[ChatMessage] encapsulation of the last agent message.
        This property checks the last message from an agent and wraps it in an Option type for
        safe handling of presence or absence of a message. If a message is present, it returns Option[ChatMessage]
        with the message, otherwise returns Option[ChatMessage] with None.
        
        Returns:
            (Option[ChatMessage]):
                 An Option object containing the last agent message or None.

        """
        return Option(self.last_agent_message)

    def last_message_from_skill(self, skill_name: str) -> Optional[ChatMessage]:
        """
        Fetches the last chat message from a particular skill.
        This method searches for the last message in the message history that originates from the specified skill.
        
        Args:
            skill_name (str):
                 The name of the skill that the message is expected to be from.
        
        Returns:
            (Optional[ChatMessage]):
                 Returns the last `ChatMessage` sent by the specified skill if found, otherwise, None.
            

        """

        def predicate(message: ChatMessage):
            """
            Checks if a ChatMessage is both of a specific kind and from a specific source.
            This function predicates if a given ChatMessage object meets two conditions: being of a specified kind (e.g., Skill) and originating from a specific source. To make the determination, it internally calls two other methods of the ChatMessage object: `is_of_kind` and `is_from_source`.
            
            Args:
                message (ChatMessage):
                     The message to be evaluated.
            
            Returns:
                (bool):
                     True if the `message` is of the kind `ChatMessageKind.Skill` and from the source `skill_name`, otherwise False.
            
            Note:
                This function assumes `ChatMessageKind.Skill` and `skill_name` are accessible within the scope and that the `ChatMessage` class has the methods `is_of_kind` and `is_from_source` implemented.

            """
            return message.is_of_kind(ChatMessageKind.Skill) and message.is_from_source(skill_name)

        return self._last_message_filter(predicate)

    def try_last_message_from_skill(self, skill_name: str) -> Option[ChatMessage]:
        """
        Attempts to retrieve the last chat message from a specified skill.
        This method checks for the last message that was sent by a particular skill and wraps the result in an 'Option' object.
        This allows the caller to handle the absence of a message safely without risking a 'NoneType' related error.
        
        Args:
            skill_name (str):
                 The name of the skill from which to retrieve the last chat message.
        
        Returns:
            (Option[ChatMessage]):
                 An 'Option' object containing the last 'ChatMessage' from the specified skill if it exists,
                or an 'Option' object containing None if no such message is present.
            

        """

        return Option(self.last_message_from_skill(skill_name))

    def _last_message_filter(self, predicate: Callable[[ChatMessage], bool]) -> Optional[ChatMessage]:
        """
        Filters the last message in the chat that meets a specific condition.
        This method applies a given predicate to messages stored in the chat in reverse order and returns the first message
        that satisfies the predicate condition. If no message satisfies the predicate, the method returns None.
        
        Args:
            predicate (Callable[[ChatMessage], bool]):
                 A function that takes a ChatMessage object as its parameter and
                returns a boolean value. It determines the condition that the ChatMessage must satisfy.
        
        Returns:
            (Optional[ChatMessage]):
                 The last message in the chat that satisfies the predicate; otherwise, None if no such
                message exists.
            (Remarks):
                The predicate function must handle ChatMessage objects. The internal typeguard_predicate is used to
                ensure that only messages of type ChatMessage are considered. The search is performed in reverse order,
                meaning it starts from the most recent message and goes backwards through the history of messages.

        """
        def typeguard_predicate(message: ChatMessage) -> TypeGuard[Optional[ChatMessage]]:
            """
            Check if the given message is an instance of `ChatMessage` and matches a specific predicate.
            This function serves as a type guard, which verifies both the type of the object and its properties according to a predicate function. If the check passes, it allows the type checking system to infer the correct type henceforth.
            
            Args:
                message (ChatMessage):
                     The message object to be checked against the type and predicate.
            
            Returns:
                (TypeGuard[Optional[ChatMessage]]):
                     `True` if `message` is an instance of `ChatMessage`
                    and matches the predicate; otherwise, `False`.
                

            """
            return isinstance(message, ChatMessage) and predicate(message)

        return first(filter(typeguard_predicate, self.reversed), None)

    @staticmethod
    def message_kind_predicate(kind: ChatMessageKind) -> Callable[[ChatMessage], bool]:
        """
        
        Returns a predicate function that determines if a ChatMessage is of the specified kind.
            This static method generates a lambda function that will check whether a given `ChatMessage`
            object matches the kind specified in the `kind` parameter. This is useful for filtering
            messages based on their type.
        
        Args:
            kind (ChatMessageKind):
                 The kind of chat message to test for.
        
        Returns:
            (Callable[[ChatMessage], bool]):
                 A function that takes a ChatMessage as its argument and returns
                True if the message is of the specified kind, False otherwise.
            

        """
        return lambda m: m.is_of_kind(kind)
