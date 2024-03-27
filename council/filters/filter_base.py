"""
Module `filter_base` provides an abstract base class for creating message filters within a chat monitoring system.

This module contains the definition of `FilterBase`, an abstract class that inherits from both
`Monitorable` and Python's built-in `ABC` (Abstract Base Class) to define a common interface for message
filters. It also includes a custom exception `FilterException` used to handle filter-specific errors.

Classes:

    FilterException(Exception):
        A custom exception used to signify errors within the filter execution process.

    FilterBase(Monitorable, ABC):
        An abstract base class defining the structure and behavior of filters.
        Filters inheriting from this class must provide implementations for the `_execute` method.

Attributes:
    None

Methods:
    execute(context: AgentContext) -> List[ScoredChatMessage]:
        A public method that serves as the entry point for filter execution. It ensures proper context
        management and calls the abstract method `_execute` which must be implemented by subclasses.

    _execute(context: AgentContext) -> List[ScoredChatMessage]:
        An abstract method that defines the specific logic to filter messages. This method should be
        overridden in concrete subclasses to implement the desired filtering functionality.



"""
from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Monitorable, ScoredChatMessage


class FilterException(Exception):
    """
    A custom exception class for handling filter-related errors.
    This exception is raised when an error occurs in the context of filtering data or operations.
    It extends the base Exception class and does not add any new functionality, but provides a more semantic error name to improve the readability and maintainability of the code.
    
    Attributes:
        message (str):
             A human-readable message indicating the error encountered.
        

    """
    def __init__(self, message: str):
        """
        Initialize the object.
        
        Args:
            message (str):
                 A description or message to be associated with the object upon initialization.
            

        """
        super().__init__(message)


class FilterBase(Monitorable, ABC):
    """
    Class representing the base for a filter mechanism in a monitoring system.
    The FilterBase class inherits from Monitorable and ABC (Abstract Base Class) and serves as a template for
    subclasses that implement specific filtering logic. It is designed to process a context and return a list
    of scored chat messages based on certain criteria.
    
    Attributes:
        None explicitly declared within FilterBase, relies on inherited attributes.
    
    Methods:
        __init__():
             Initializes a new instance of the FilterBase class.
        execute(context):
             Wraps the filtering process in a context manager before calling the internal _execute method.
        _execute(context):
             Abstract method that must be implemented by subclasses to define the actual filtering process.
            Subclasses are required to override and implement the _execute method while adhering to the expected input and
            output signature.
        Inherits From:
            Monitorable
            ABC (Abstract Base Class)

    """

    def __init__(self):
        """
        Initializes a new instance of the class.
        This constructor initializes the instance with a specific filter name by
        calling the superclass constructor with 'filter' as an argument.
        
        Args:
            None
        
        Returns:
            None

        """
        super().__init__("filter")

    def execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Executes a given action within a specific agent context and returns a list of scored chat messages.
        This method is responsible for contextual execution of a task, where it leverages the provided
        `AgentContext` to perform necessary pre- and post-execution steps automatically using a context
        manager. The actual execution logic is implemented in the `_execute` method, which is called
        within this context.
        
        Args:
            context (AgentContext):
                 The context in which the action is to be executed, providing all
                the necessary information and tools required for the execution of the action.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of chat messages, each associated with a score, as a result
                of the action execution within the given context.

        """
        with context:
            return self._execute(context=context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Abstract method to be implemented by subclasses to execute an action within a given context.
        This method should contain the logic for performing a specific action that takes the current
        `AgentContext` as input and produces a list of `ScoredChatMessage` objects, which represent
        messages with associated scores based on relevance or other scoring criteria defined by the implementation.
        
        Args:
            context (AgentContext):
                 The context in which the action should be executed. This object contains
                environment state, historical data, and other relevant information that the action might need.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of `ScoredChatMessage` instances, each containing a chat message
                and a corresponding score indicating the message's relevance or priority.
        
        Raises:
            NotImplementedError:
                 If the method is not implemented in a subclass, invoking it will
                raise a `NotImplementedError` to indicate that the concrete subclass must provide
                its own implementation of the method.

        """
        pass
