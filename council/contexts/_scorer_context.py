"""

Module `_scorer_context` - Provides context management for scoring processes within an agent's execution.

This module defines the `ScorerContext` class which inherits from `ContextBase` and is specialized for the scoring stage of an agent's execution
flow. It facilitates access to the agent context store, execution context, and budget specific to the scoring phase.

Classes:
    ScorerContext(ContextBase): Encapsulates the context required for scoring operations in an iterative execution process. It
    extends `ContextBase` with functionality specific to the scoring phase.

Functions:
    ScorerContext.from_context(context: ContextBase, monitored: Monitored, budget: Optional[Budget]=None) -> ScorerContext
        Factory method to create a new `ScorerContext` from an existing `ContextBase` instance, replacing the monitored item and
        the budget if provided.

    ScorerContext.empty() -> ScorerContext
        Factory method to create a new `ScorerContext` with default values for `AgentContextStore`, `ExecutionContext`, and initializes
        with an `InfiniteBudget`, representing a scoring context with no restrictions.

    ScorerContext.new_for(self, monitored: Monitored) -> ScorerContext
        Creates a new `ScorerContext` configured for a specific monitored item using the current context's store and execution
        context, while retaining the original budget. This is useful for creating nested scoring contexts for a hierarchical
        monitoring approach.


"""
from __future__ import annotations

from typing import Optional

from ._agent_context_store import AgentContextStore
from ._budget import Budget, InfiniteBudget
from ._chat_history import ChatHistory
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._monitored import Monitored


class ScorerContext(ContextBase):
    """
    A context class that provides additional functionality for scoring purposes.
    The ScorerContext class extends from ContextBase and is designed to handle
    the context required for scoring processes in an agent-based system. It
    allows the creation of a new scoring context based on existing data or from
    scratch, facilitating the adjustment of the context to changing scoring
    targets (monitored objects) and budget constraints.
    
    Attributes:
        Inherits all attributes from the ContextBase class.
    
    Methods:
        __init__(store, execution_context, budget):
            Initializes a new instance of ScorerContext.
    
    Args:
        store (AgentContextStore):
             The store holding agent context data.
        execution_context (ExecutionContext):
             The current execution context.
        budget (Budget):
             The budget for the scoring context.
        from_context(context, monitored, budget):
            Creates a ScorerContext based on a given context with modifications
            for a new monitored object and optionally a new budget.
    
    Args:
        context (ContextBase):
             The original context to base the new context on.
        monitored (Monitored):
             The new object to be monitored in the scoring process.
        budget (Optional[Budget]):
             An optional new budget to override the original context's budget.
    
    Returns:
        (ScorerContext):
             A new ScorerContext object tailored to the provided monitored object and budget.
        empty():
            Constructs a ScorerContext with default empty values for its attributes.
    
    Returns:
        (ScorerContext):
             An instance of ScorerContext with default empty attributes.
        new_for(monitored):
            Creates a new ScorerContext for a different monitored object using the current ScorerContext's data.
    
    Args:
        monitored (Monitored):
             The new object to be monitored in the new scoring context.
    
    Returns:
        (ScorerContext):
             A new ScorerContext instance dedicated to the provided monitored object.

    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        """
        Initializes a new instance of the class with the given store, execution context, and budget.
        
        Args:
            store (AgentContextStore):
                 The storage for agent-related contextual information.
            execution_context (ExecutionContext):
                 The context in which the agent will execute its tasks.
            budget (Budget):
                 The budget allocated for the agent's operations.
            

        """
        super().__init__(store, execution_context, budget)

    @staticmethod
    def from_context(context: ContextBase, monitored: Monitored, budget: Optional[Budget] = None) -> ScorerContext:
        """
        Creates a new ScorerContext based on an existing context, but with a new execution context for a specific monitored entity and a potentially different budget.
        This method is a factory function that facilitates the creation of a ScorerContext object derived from another base context. It utilizes the new_for method of ExecutionContext to create a new execution context specific to the monitored entity provided. If a budget is not supplied, the new ScorerContext will inherit the budget of the base context.
        
        Args:
            context (ContextBase):
                 The base context from which to create a new ScorerContext.
            monitored (Monitored):
                 The monitored entity for which the new execution context will be created.
            budget (Optional[Budget]):
                 An optional budget to assign to the new context. If not provided, the base context's budget is used.
        
        Returns:
            (ScorerContext):
                 A new ScorerContext object with the new execution context and the specified or inherited budget.

        """
        return ScorerContext(context._store, context._execution_context.new_for(monitored), budget or context._budget)

    @staticmethod
    def empty() -> ScorerContext:
        """
        Creates a new instance of the ScorerContext class with default empty values.
        This static method returns a new ScorerContext object where all the components are initialized but empty. The AgentContextStore is initialized with an empty ChatHistory, the ExecutionContext with default values, and an InfiniteBudget is used for the budget.
        
        Returns:
            (ScorerContext):
                 A new instance of ScorerContext with empty/default initialized components.

        """
        return ScorerContext(AgentContextStore(ChatHistory()), ExecutionContext(), InfiniteBudget())

    def new_for(self, monitored: Monitored) -> ScorerContext:
        """
        Creates a new ScorerContext for a specified monitored object by utilizing the current context.
        This method acts as a factory for creating a new instance of ScorerContext specific to the monitored object provided.
        It allows inheriting properties and configurations from the current context to maintain consistency across different
        ScorerContext instances used for various monitored objects.
        
        Args:
            monitored (Monitored):
                 The object that needs monitoring and will be associated with the created ScorerContext.
        
        Returns:
            (ScorerContext):
                 A new instance of ScorerContext configured for the provided monitored object.

        """
        return self.from_context(self, monitored)
