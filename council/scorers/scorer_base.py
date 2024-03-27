"""


Module scorer_base

This module defines the abstract base class for scoring mechanisms, handling
the common scoring operations and exception handling. It also contains necessary
import statements and any other base functionality required for the scoring process.

Classes:
    ScorerBase(Monitorable, abc.ABC): An abstract base class for implementing different scoring algorithms.

Exceptions:
    ScorerException: A custom exception that represents errors occurring within the scoring process.



"""
import abc
from typing import Any, Dict

from council.contexts import ChatMessage, Monitorable, ScorerContext
from .scorer_exception import ScorerException


class ScorerBase(Monitorable, abc.ABC):
    """
    A base class for creating scoring entities that evaluate messages within a given context.
    This abstract class serves as a template for creating specific scorers that assess ChatMessage
    data based on some criteria. Implementers must define the `_score` method, which is the core of
    each scorer. The class inherits from `Monitorable` and uses the abstract base class (abc) module
    to enforce the definition of the `_score` method in subclasses.
    
    Attributes:
        Inherited from Monitorable but not explicitly defined in this class.
    
    Methods:
        __init__(self):
            Initializes the ScorerBase instance by calling the constructor of the superclass
            `Monitorable` with the identifier 'scorer'.
        score(self, context:
             ScorerContext, message: ChatMessage) -> float:
            Public method that safely executes the scoring process by calling the internal `_score` method.
            It handles any exceptions that occur during the scoring, logs them, and raises a ScorerException.
    
    Args:
        context (ScorerContext):
             The context in which the scoring is performed, providing
            relevant information and utilities needed for assessing the message.
        message (ChatMessage):
             The chat message to be scored.
    
    Returns:
        (float):
             The score assigned to the message by the internal `_score` method.
    
    Raises:
        ScorerException:
             An exception that is raised when an error occurs during the
            execution of the `_score` method.
        _score(self, context:
             ScorerContext, message: ChatMessage) -> float:
            An abstract method that must be implemented by subclasses. This method contains the
            actual scoring logic for evaluating a message within the given context.
    
    Args:
        context (ScorerContext):
             The scoring context containing necessary information
            for score computation.
        message (ChatMessage):
             The message object that is being evaluated.
    
    Returns:
        (float):
             The computed score for the message.
        to_dict(self) -> Dict[str, Any]:
            Serializes the scorer type information into a dictionary.
    
    Returns:
        (Dict[str, Any]):
             A dictionary with a single key 'type' that maps to the class name
            of the scorer instance.
        

    """

    def __init__(self):
        """
        Initializes an object of the class.
        This method is the constructor for the class. It initializes the object with the name 'scorer'.
        The constructor calls the superclass's __init__ method with the argument 'scorer' to establish the
        identity or functionality specific to this subclass.
        

        """
        super().__init__("scorer")

    def score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        Computes the score for a given message within a scoring context.
        This method wraps around the private '_score' method to handle the actual scoring logic.
        If scoring succeeds, it returns a floating-point score. If an exception occurs,
        it logs the exception message and re-raises a ScorerException with the original exception nested within.
        
        Args:
            context (ScorerContext):
                 The context for scoring that provides relevant data and functionality.
            message (ChatMessage):
                 The chat message that is being scored.
        
        Returns:
            (float):
                 The computed score for the provided chat message.
        
        Raises:
            ScorerException:
                 An exception raised when scoring cannot be completed due to
                an error in the '_score' method. The original exception
                is nested within the ScorerException.
            

        """
        try:
            return self._score(context, message)
        except Exception as e:
            context.logger.exception('message="execution failed"')
            raise ScorerException from e

    @abc.abstractmethod
    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        """
        Abstract method to calculate a score for a chat message within a given context.
        This method should be implemented by subclasses, and it should provide a way to
        assign a numerical score to a message in a chat, which can be used to rank,
        filter, or otherwise evaluate messages. The scoring may depend on various
        aspects of the context and the message itself.
        
        Args:
            context (ScorerContext):
                 An instance of ScorerContext that provides context
                such as the conversation history, the user profile, the time of the message,
                and other relevant data that might influence the scoring process.
            message (ChatMessage):
                 The chat message instance to be scored. This object
                will typically contain the content of the message along with any metadata
                such as the sender, timestamp, etc.
        
        Returns:
            (float):
                 A floating-point number representing the calculated score for the
                given message within the context provided.

        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the instance to a dictionary representation.
        This method serializes the instance by creating a dictionary with a key 'type' that holds the name of the instance's class. It helps in identifying the object type when the instance needs to be serialized, for instance, for potential JSON conversion or for debugging purposes.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary with the class name of the instance under the key 'type'.

        """
        return {"type": self.__class__.__name__}
