"""

Module controller_base

This module defines the abstract base class ControllerBase along with the accompanying ControllerException
used within a system managing execution units for different chains.
The ControllerBase class serves as an abstract base class for building controllers that handle the execution
process of chains in a certain context. It integrates with the Monitorable class to enable monitoring
of the controller's activities.

Classes:
    ControllerException: A custom exception class for handling errors specific to the ControllerBase
    and its derived classes.
    ControllerBase: An abstract base class that defines the structure and enforces the implementation
    of the control logic for managing execution units.

Imports:
    ABC: Abstract base class module from the 'abc' library used for creating abstract base classes.
    abstractmethod: Decorator indicating abstract methods within an abstract base class.
    List, Sequence, Optional: Types from the 'typing' module for type hinting.
    ChainBase: Custom class imported from the 'council.chains' module representing the base chain.
    AgentContext: Custom class imported from the 'council.contexts' module representing the agent's context.
    Monitorable: Custom class which ControllerBase extends to include monitoring functionality.
    ExecutionUnit: Custom class imported from the '.execution_unit' module representing an execution unit.

Attributes:
    ControllerException has a single attribute 'message' which is a string containing the exception message. 
    ControllerBase has two attributes:
        - _chains (List[ChainBase]): A list of chains that the controller manages.
        - _parallelism (bool): A boolean flag indicating if parallel execution of chains is enabled.

Methods:
    ControllerBase.execute(): A method that executes the control logic in the given context and returns a list
    of ExecutionUnit objects spawned as a result.
    ControllerBase._execute(): An abstract method that must be implemented by derived classes to define the concrete
    control logic for execution.

Properties:
    ControllerBase.chains: A property that returns the list of chains managed by the controller.
    ControllerBase.default_execution_unit_rank: A property that shows the default rank for execution units
    which is dependent on parallelism capability.


"""
from abc import ABC, abstractmethod
from typing import List, Sequence, Optional

from council.chains import ChainBase
from council.contexts import AgentContext, Monitorable
from .execution_unit import ExecutionUnit


class ControllerException(Exception):
    """
    A custom exception class for handling controller-specific errors.
    This exception class is intended to be used when raising errors that are
    specific to the operations of a controller within an application. The
    ControllerException class extends the base Exception class and is
    initialized with an error message that can be used to provide more
    detailed information about the exception that occurred.
    
    Attributes:
        message (str):
             A human-readable message describing the error.
        

    """
    def __init__(self, message: str):
        """
        Initialize a new instance of the class.
        The constructor initializes the object with a provided message that describes the exception.
        It calls the parent class's initializer with the message provided.
        
        Args:
            message (str):
                 The message string describing the exception.
            

        """
        super().__init__(message)


class ControllerBase(Monitorable, ABC):
    """
    Base abstract class for creating controller objects that manage execution of chains.
    A controller inherits from the Monitorable class and abstract base class (ABC) indicating that it
    is intended to be a base class for other classes without being instantiated on its own.
    Each controller manages a sequence of 'ChainBase' instances allowing for execution management
    in either a parallel or sequential manner based on the 'parallelism' flag.
    
    Attributes:
        _chains (List[ChainBase]):
             A private list of chain instances that the controller will execute.
        _parallelism (bool):
             A private boolean indicating whether the chains should be executed in parallel.
    
    Args:
        chains (Sequence[ChainBase]):
             A sequence of chain instances that the controller will manage.
        parallelism (bool, optional):
             A flag indicating whether chain execution should be parallel. Defaults to False.
    
    Methods:
        execute(context:
             AgentContext) -> List[ExecutionUnit]: Public method to execute the chains using
            the given context. It automatically manages the context resource.
        _execute(context:
             AgentContext) -> List[ExecutionUnit]: Abstract internal method defined to
            be implemented by subclasses for the actual execution logic of the controller.
        Properties:
        chains (Sequence[ChainBase]):
             Public property to access the sequence of chains.
        default_execution_unit_rank (Optional[int]):
             Public property to get the default rank of execution
            units based on whether parallelism is enabled or not.

    """

    def __init__(self, chains: Sequence[ChainBase], parallelism: bool = False):
        """
        Initialize the Controller class with a sequence of ChainBase instances and a parallelism flag.
        
        Args:
            chains (Sequence[ChainBase]):
                 A sequence of ChainBase instances that this controller will manage.
            parallelism (bool, optional):
                 A flag indicating whether parallelism is enabled or not. Defaults to False.
                The initializer stores a list of chains and sets the parallelism attribute according to the
                provided parallelism argument.

        """
        super().__init__("controller")
        self._chains = list(chains)
        self._parallelism = parallelism

    def execute(self, context: AgentContext) -> List[ExecutionUnit]:
        """
        Executes the function within the given context and returns a list of ExecutionUnits.

        """
        with context:
            return self._execute(context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        """
        Abstract method that when implemented should execute actions within a given context.
        This method is expected to be overridden in a subclass and must provide a mechanism to perform
        a series of actions or calculations based on the agent's context. The result of these actions
        is a list of ExecutionUnit objects, which encapsulate the execution details that have been produced
        by the method.
        
        Args:
            context (AgentContext):
                 An instance of AgentContext that provides necessary information
                and interfaces for the execution of the method.
        
        Returns:
            (List[ExecutionUnit]):
                 A list of ExecutionUnit instances which represent the outcome of
                the execution process.
        
        Raises:
            NotImplementedError:
                 If the method is not implemented in a subclass. This is because
                _execute is an abstract method and is intended to be defined by
                the inheriting class.

        """
        pass

    @property
    def chains(self) -> Sequence[ChainBase]:
        """
        Retrieves the chains associated with the current object.
        This property provides access to the chains which are a sequence of `ChainBase` objects connected to the current instance. It simply returns the protected member that holds the collection of chains.
        
        Returns:
            (Sequence[ChainBase]):
                 A sequence containing the chain objects.
            

        """
        return self._chains

    @property
    def default_execution_unit_rank(self) -> Optional[int]:
        """
        
        Returns the default rank of execution units based on parallelism attribute.
            This property method checks if the object's _parallelism attribute is truthy. If it is,
            it returns 1, indicating a default rank. If _parallelism is falsy, it returns None to
            signify that there is no default rank.
        
        Returns:
            (Optional[int]):
                 An integer value of 1 if parallelism is enabled, otherwise, None.

        """
        return 1 if self._parallelism else None
