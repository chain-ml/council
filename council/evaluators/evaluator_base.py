"""

Module for the base structure and exception definition for Evaluators.

This module defines an abstract base class `EvaluatorBase` for implementing different kinds of evaluators,
which are components responsible for scoring or ranking chat messages within a given context.
An evaluator may integrate with various subsystems and produce a list of `ScoredChatMessage`
objects that represent the messages and their corresponding scores.

Classes:
    EvaluatorException: Custom exception class used within the Evaluator framework.
    EvaluatorBase: An abstract base class that defines the standard interface and behavior for
                   all evaluators. It inherits from `Monitorable` and `ABC` to provide monitoring
                   capabilities and to enforce the implementation of abstract methods,
                   respectively.

Exceptions:
    EvaluatorException: Raised when an error occurs within the evaluating process.



"""
from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Monitorable, ScoredChatMessage


class EvaluatorException(Exception):
    """
    
    Raises an exception for errors specific to the evaluator module.
        This custom exception class inherits from the built-in Exception class and is used to indicate
        errors that are specific to the evaluation process within the application. Instances of this
        class should be thrown when an evaluator encounters an issue that cannot be recovered from
        and requires the attention of the calling code to handle.
    
    Attributes:
        message (str):
             A human-readable message describing the error that occurred.
    
    Args:
        message (str):
             A human-readable message describing the error that occurred.
        Inherits:
        Exception:
             Inherent methods and properties of the Exception class are available to EvaluatorException.

    """
    def __init__(self, message: str):
        """
        Initialize a new instance of the class.
        This constructor initializes the object with a specific message. It calls
        the base class constructor with the message. This is typically used when
        creating an object of a class that is derived from an exception class,
        allowing for custom messages to be associated with the generated exception.
        
        Args:
            message (str):
                 A descriptive message associated with the exception.
            

        """
        super().__init__(message)


class EvaluatorBase(Monitorable, ABC):
    """
    A base class for creating evaluator objects in a messaging context, with an abstract method for actual evaluation execution.
    This abstract base class defines a generic evaluator which relies on a provided AgentContext to process and score chat messages. It inherits from Monitorable to enable monitoring capabilities and from the ABC module to ensure it cannot be instantiated directly. Implementers should provide specific logic for the '_execute' method, which carries out the evaluation process.
    
    Attributes:
        None
    
    Methods:
        __init__:
             Constructs the base evaluator, initializing its type as 'evaluator'.
        execute:
             Public method to initiate the evaluation process and log the operation.
            It takes a context (AgentContext) and returns a list of ScoredChatMessage instances.
            This method wraps the _execute call with a logging context manager.
        _execute:
             An abstract method that must be implemented by subclasses to perform
            the actual evaluation process on the provided context. It is expected to
            return a list of ScoredChatMessages.
    
    Note: Subclasses must override the _execute method providing the specific logic
        for message evaluation and scoring within their domain.

    """

    def __init__(self):
        """
        Initializes a new instance of the class which is assumed to be an "evaluator" type object.
        This constructor invokes the initializer of the superclass with the specific argument 'evaluator' to establish the type or role of the instance being created.
        
        Raises:
            Any exceptions raised by the superclass initializer will be propagated as this method does not contain any exception handling.

        """
        super().__init__("evaluator")

    def execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Executes a given context within a logging scope.
        This method is intended to be run within an AgentContext's log entry context manager, which handles logging around the execution. The actual execution logic is deferred to the '_execute' method of the object, which is expected to be implemented by the subclass where this method is included.
        
        Args:
            context (AgentContext):
                 The context in which the agent will operate, containing state and environment information that the execute function can use to perform its task.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of scored chat messages generated from the execution of the '_execute' method within the provided context. ScoredChatMessage is assumed to be a data structure that holds a chat message along with its associated score.

        """
        with context.log_entry:
            return self._execute(context=context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Executes an operation within a given agent context and returns a list of scored chat messages.
        This method is an abstract method meant to be implemented by subclasses. The implementation should define the logic to perform a specific task that uses the provided `AgentContext`. The task will typically produce a list of messages that are relevant to the context, each accompanied by a score indicating the message's relevance or importance.
        
        Args:
            context (AgentContext):
                 An instance of `AgentContext` that provides the execution context and necessary data for the operation.
        
        Returns:
            (List[ScoredChatMessage]):
                 A list of `ScoredChatMessage` objects. Each object in the list should contain a chat message and an associated score representing the message's relevance or quality in the given context.

        """
        pass
