"""

Module for abstract ChainBase class defining a chain execution flow.

This module defines an abstract base class ChainBase, which provides the core structure and functionality for a chain of operations, potentially within a larger workflow. The ChainBase class inherits from the Monitorable class and the abstract base class abc.ABC, ensuring that it can be monitored and requires concrete implementations of its abstract methods for execution. It contains methods for getting the chain's metadata, such as its name and description, and whether it supports instructions. It also contains a critical method for executing the chain within a given context.

Classes:
    ChainBase(Monitorable, abc.ABC): An abstract base class for defining chains.

    This class initializes chain instances with name, description, and instruction support flag. It also provides properties to access these attributes. The execute method initializes the context manager and calls the internal _execute method, which subclasses must implement. Overridden __repr__ and __str__ methods allow for a more readable string representation of the ChainBase class instances.

Attributes:
    _name (str): The name of the chain.
    _description (str): A brief description of the chain's purpose or functionality.
    _instructions (bool): Flag indicating if the chain supports operational instructions.



"""
import abc
from typing import Optional

from council.contexts import ChainContext, Monitorable
from council.runners import RunnerExecutor


class ChainBase(Monitorable, abc.ABC):
    """
    Class that serves as a base for chain-like structures that can be monitored.
    This abstract class provides a common interface and basic functionality
    for different types of chains, which are meant to have a sequential or
    linked nature, and can perform a series of operations when executed.
    It also inherits from `Monitorable` to allow monitoring of the chain's
    activities.
    
    Attributes:
        _name (str):
             Name of the chain.
        _description (str):
             A brief description of the chain's purpose and functionality.
        _instructions (bool):
             Flag to indicate whether the chain supports instructions.
    
    Args:
        name (str):
             The name assigned to this chain instance.
        description (str):
             The description of this chain instance.
        support_instructions (bool, optional):
             Flag indicating if the chain supports
            instructions. Defaults to False.
    
    Methods:
        name:
             Property that returns the name of the chain.
        description:
             Property that returns the description of the chain.
        is_supporting_instructions:
             Property that returns whether the chain supports
            instructions.
        execute:
             Executes the chain with a given context and an optional executor.
            This function wraps the execution with the chain's context to
            ensure proper setup/teardown and delegates the actual execution
            to the `_execute` method.
        _execute:
             An abstract method that subclasses must implement to define
            the chain's execution logic. It is called by the `execute` method.
    
    Returns:
        A string representation of the chain when the `__repr__` or `__str__` methods
        are called.
    
    Note:
        The `ChainContext` and `RunnerExecutor` types should be defined elsewhere
        within the codebase, as they are used as arguments for the execute functions.
        The ChainBase cannot be instantiated directly as it is an abstract class; it requires
        subclasses to provide specific implementations of the `_execute` method.

    """

    _name: str
    _description: str
    _instructions: bool

    def __init__(self, name: str, description: str, support_instructions: bool = False):
        """
        Initializes a new instance of the class.
        This constructor initializes the instance by setting its name, description, and optionally whether it supports instructions.
        It also sets the monitor's name to the provided name.
        
        Args:
            name (str):
                 The name to assign to this instance.
            description (str):
                 The description to assign to this instance.
            support_instructions (bool, optional):
                 Determines if instructions are supported. Defaults to False.
            

        """
        super().__init__("chain")
        self._name = name
        self._description = description
        self._instructions = support_instructions
        self.monitor.name = name

    @property
    def name(self) -> str:
        """
        Gets the name attribute of the instance.
        This @property-decorated function acts as a getter for the instance's `_name` attribute,
        allowing a controlled access to retrieve the value of `_name`. Calling this function
        returns the current value of the private `_name` attribute without directly accessing it.
        
        Returns:
            (str):
                 The current value of the `_name` attribute of the instance.

        """
        return self._name

    @property
    def description(self) -> str:
        """
        Property that retrieves the description of an object.
        This property getter method returns the private '_description' attribute of the object, allowing
        read-only access to the description.
        
        Returns:
            (str):
                 A string containing the description of the object.

        """
        return self._description

    @property
    def is_supporting_instructions(self) -> bool:
        """
        
        Returns whether the object supports instructions.
            This property method checks the object to determine if it has non-empty instructions.
            It is an indicator that can be used to determine if the object is capable of providing instructions.
        
        Returns:
            (bool):
                 True if the object has instructions, False otherwise.
            

        """
        return self._instructions

    def execute(self, context: ChainContext, executor: Optional[RunnerExecutor] = None) -> None:
        """
        Executes a process using a given context and an optional executor.
        This method makes use of the provided `ChainContext` to execute a process which is handled by the `_execute` method.
        An optional `RunnerExecutor` can be used to override the default execution. The context is managed by a `with` statement
        to ensure proper acquisition and release of resources during execution.
        
        Args:
            context (ChainContext):
                 The context in which the process should be executed. This must be an instance of `ChainContext`.
            executor (Optional[RunnerExecutor]):
                 An optional executor that can be provided to handle the process execution.
                If no executor is provided, the execution is handled by the default executor.
        
        Raises:
            Exception:
                 If any exception occurs during the execution of the process, it is raised to the caller of this method.
            

        """
        with context:
            self._execute(context, executor)

    @abc.abstractmethod
    def _execute(
        self,
        context: ChainContext,
        executor: Optional[RunnerExecutor] = None,
    ) -> None:
        """
        Performs execution of a command within a given context using an optional executor.
        This method is abstract and must be implemented by subclasses.
        It outlines the steps that need to be carried out during the execution phase of a process.
        
        Args:
            context (ChainContext):
                 The context within which the command execution takes place.
                Contains all the environment and state information required for execution.
            executor (Optional[RunnerExecutor]):
                 An optional executor that can be used to run the command.
                If not provided, some default mechanism of execution should be utilized. The executor
                encapsulates the logic and resources that drive the execution of the command.
        
        Returns:
            (None):
                 This method is intended to perform an action and thus does not return any value.

        """
        pass

    def __repr__(self):
        """
        Represents the Chain's object string representation method.
        This special method is called by the repr() built-in function and by string conversions (reverse quotes) to compute the "official" string representation of an object. If at all possible, this should look like a valid Python expression that could be used to recreate an object with the same value (given an appropriate environment). If this is not possible, a string of the form <...some useful description...> should be returned. The return value must be a string object. If a class defines __repr__() but not __str__(), then __repr__() is also used when an “informal” string representation of instances of that class is required.
        This is typically used for debugging, so it is important that the representation is information-rich and unambiguous.
        
        Returns:
            (str):
                 The string representation of the Chain instance, formatted in a way that is potentially executable as a Python expression usually in the form 'Chain(name, description)'.

        """
        return f"Chain({self.name}, {self.description})"

    def __str__(self):
        """
        Return a string representation of the chain instance with its name and description included. This magic method overrides the default behavior to provide a readable string display when the object is printed or converted to a string in other contexts. The returned string includes the instance's `name` and `description` attributes, formatted in a human-friendly way. This can be used for logging, debugging, or simply to present information to the end user. Returns: str: A formatted string, which includes the `name` and `description` of the chain instance.

        """
        return f"Chain {self.name}, description: {self.description}"
