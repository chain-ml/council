"""

Module that defines the skill context and related functionality for chaining iterations within a skill execution flow.

This module provides the necessary classes to store and access the context information needed for the execution of skills, which includes managing message histories, iteration contexts, execution logging, and resource budgeting.

Classes:
    IterationContext: Represents the context for a particular iteration with an associated index and value.
    SkillContext: A specialized subclass of ChainContext that includes additional information about the current iteration.

Functions:
    SkillContext.empty: Static method that returns an Option representing an absent IterationContext.
    SkillContext.new: Static method that creates a new IterationContext wrapped in an Option, given an index and a value.
    SkillContext.from_chain_context: Static method that creates a SkillContext from an existing ChainContext, including the current iteration context.

Typical usage example:

    iteration_ctx = IterationContext.new(index=1, value=some_value)
    skill_ctx = SkillContext.from_chain_context(chain_context=existing_chain_context, iteration=iteration_ctx)


"""
from __future__ import annotations

from typing import Any, Iterable

from council.utils import Option
from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chain_context import ChainContext
from ._chat_message import ChatMessage
from ._execution_context import ExecutionContext


class IterationContext:
    """
    A context container representing the state of an iteration, encapsulating an index and a value.
    This class is designed to hold the current state of an iterative process, including
    an index indicating the position within the iteration, and the value at that position.
    It provides methods to instantiate empty or populated instances of the context.
    
    Attributes:
        _index (int):
             A private attribute to store the index of the iteration.
        _value (Any):
             A private attribute to hold the value at the current index of the iteration.
    
    Methods:
        __init__:
             Constructs an IterationContext with a given index and value.
        index:
             Property to get the current index of the iteration context.
        value:
             Property to get the current value of the iteration context.
        empty:
             Static method that returns an empty Option of IterationContext.
        new:
             Static method that creates a new IterationContext with a given index and value, wrapped in an Option.
    
    Returns:
        (IterationContext):
             An instance of IterationContext.

    """

    def __init__(self, index: int, value: Any) -> None:
        """
        Initializes a new instance of the class with specified index and value attributes.
        
        Args:
            index (int):
                 The index associated with the value.
            value (Any):
                 The value to be stored, can be of any type.
        
        Attributes:
            _index (int):
                 A private attribute that stores the index.
            _value (Any):
                 A private attribute that stores the value.
        
        Returns:
            (None):
                 This method does not return a value, it only initializes the object.

        """
        self._index = index
        self._value = value

    @property
    def index(self) -> int:
        """
        
        Returns the index attribute of the instance.
            Property method that gets the current value of the instance's _index attribute.
            This attribute typically represents a positional index in a collection or list.
        
        Returns:
            (int):
                 The current value of the _index attribute of the instance.

        """
        return self._index

    @property
    def value(self) -> Any:
        """
        Property that gets the current value of the object.
        This property allows accessing the value of the object in a managed way, providing
        a layer of abstraction over direct access to the underlying value attribute.
        
        Returns:
            (Any):
                 The current value held in the '_value' attribute of the object.
            

        """
        return self._value

    @staticmethod
    def empty() -> Option["IterationContext"]:
        """
        
        Returns an `Option` with a none value for `IterationContext`.
            This static method provides a way to represent a lack of an `IterationContext` by returning an `Option` with a none value.
        
        Returns:
            (Option[`IterationContext`]):
                 An `Option` indicating the absence of a value.

        """
        return Option.none()

    @staticmethod
    def new(index: int, value: Any) -> Option["IterationContext"]:
        """
        Creates a new `IterationContext` object encapsulated within an `Option`.
        This static method constructs a new `IterationContext` by taking an index and a value, then
        wraps it in an `Option` object which provides a way to handle the presence or absence of a value.
        
        Args:
            index (int):
                 The index to associate with the `IterationContext`.
            value (Any):
                 The value to be included in the `IterationContext`.
        
        Returns:
            (Option[IterationContext]):
                 An `Option` object containing the newly created `IterationContext`.

        """
        return Option.some(IterationContext(index, value))


class SkillContext(ChainContext):
    """
    A context class that extends ChainContext, providing additional attributes and methods needed for skills execution within an agent context.
    The SkillContext class holds a reference to iteration-specific information alongside the inherited attributes from ChainContext, which includes the agent's store, execution context, skill name, available budget, and chat messages. SkillContext simplifies the handling of the execution context and related data required during the skill lifecycle.
    
    Attributes:
        _iteration (Option[IterationContext]):
             Optional container for iteration-related context. Protected attribute that should not be modified directly.
    
    Methods:
        __init__(self, store, execution_context, name, budget, messages, iteration):
            Initializes a new instance of the SkillContext class.
    
    Args:
        store (AgentContextStore):
             The storage for context related to the agent.
        execution_context (ExecutionContext):
             The context for the current execution state.
        name (str):
             The name of the skill.
        budget (Budget):
             The budget allocated for the skill's execution.
        messages (Iterable[ChatMessage]):
             The collection of chat messages related to the skill.
        iteration (Option[IterationContext]):
             Optional context for the current iteration.
    
    Returns:
        None.
        @property
        iteration(self) -> Option[IterationContext]:
            Property that returns iteration context. Read-only.
    
    Returns:
        (Option[IterationContext]):
             The iteration context if available; otherwise, None.
            @staticmethod
        from_chain_context(context, iteration) -> SkillContext:
            Static method that constructs a SkillContext object from a given ChainContext instance and iteration context.
    
    Args:
        context (ChainContext):
             The ChainContext instance from which to create a SkillContext.
        iteration (Option[IterationContext]):
             Optional context for the current iteration.
    
    Returns:
        (SkillContext):
             A new SkillContext instance initialized with the given ChainContext attributes and iteration context.
        

    """

    def __init__(
        self,
        store: AgentContextStore,
        execution_context: ExecutionContext,
        name: str,
        budget: Budget,
        messages: Iterable[ChatMessage],
        iteration: Option[IterationContext],
    ) -> None:
        """
        Initializes a new instance of the class with specific context information and parameters.
        This constructor sets up an environment with references to context stores, an execution context, optional iteration details, and additional attributes like the agent's name and budget.
        
        Args:
            store (AgentContextStore):
                 A storage object for agent context which provides handles to other components.
            execution_context (ExecutionContext):
                 The current state of the execution context in which the agent operates.
            name (str):
                 Human-readable identifier for this particular instance.
            budget (Budget):
                 An object encapsulating the financial constraints or allowances for the agent.
            messages (Iterable[ChatMessage]):
                 A collection of chat messages that the agent might have to process or refer to.
            iteration (Option[IterationContext], optional):
                 Optional iteration context providing additional information about the current or a specific iteration state.
            

        """
        super().__init__(store, execution_context, name, budget, messages)
        self._iteration = iteration

    @property
    def iteration(self) -> Option[IterationContext]:
        """
        Gets the current iteration context if available.
        
        Returns:
            (Option[IterationContext]):
                 An optional `IterationContext` object that represents the current iteration context.
                If the context is not available, it returns `None`.

        """
        return self._iteration

    @staticmethod
    def from_chain_context(context: ChainContext, iteration: Option[IterationContext]) -> SkillContext:
        """
        Creates a new instance of SkillContext using an existing ChainContext and an IterationContext.
        This static method takes a ChainContext object and an optional IterationContext to
        initialize a SkillContext object. The SkillContext object carries all the information
        from the ChainContext along with the additional IterationContext which provides iteration-specific data.
        
        Args:
            context (ChainContext):
                 The ChainContext instance from which to create the SkillContext.
            iteration (Option[IterationContext]):
                 An optional IterationContext instance representing the current iteration context.
        
        Returns:
            (SkillContext):
                 A new SkillContext instance built from the provided ChainContext and IterationContext.

        """
        return SkillContext(
            context._store,
            context._execution_context,
            context._name,
            context.budget,
            context.current.messages,
            iteration,
        )
