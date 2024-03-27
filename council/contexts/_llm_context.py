"""

Module _llm_context.

This module defines the context for language model (LLM) interactions within an
agent context. It extends the basic context functionalities to cater specifically
to the requirements and operations associated with LLM. It includes the definition
for LLMContext, a class that manages the agent context store, execution context,
and budgeting for LLM operations.

Classes:
    LLMContext -- Extends ContextBase and provides context management for the
                   LLM within the agent's execution context.

Functions:
    LLMContext.from_context(context: ContextBase, monitored: Monitored,
                          budget: Optional[Budget]=None) -> LLMContext

        Creates an LLMContext instance derived from an existing ContextBase
        instance and a Monitored object. Optionally, a custom Budget may be
        provided, otherwise the budget from the given context will be used.

    LLMContext.empty() -> LLMContext

        Generates an empty LLMContext with a default AgentContextStore containing
        a new ChatHistory(), a new ExecutionContext, and an InfiniteBudget instance.

    LLMContext.new_for(monitored: Monitored) -> LLMContext

        Creates a new LLMContext for the given Monitored instance, providing a
        new context that is nested under the current context and linked to the
        specified Monitored object.


"""
from __future__ import annotations
from typing import Optional

from ._agent_context_store import AgentContextStore
from ._budget import Budget, InfiniteBudget
from ._chat_history import ChatHistory
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._monitored import Monitored


class LLMContext(ContextBase):
    """
    A context handler class for Large Language Model (LLM) agents, extending a base context class.
    This class manages context to interface with a higher level representation of the agent's understanding,
    and to manipulate the context when interacting with various components and services.
    
    Attributes:
        Not applicable for class level docstring; Attributes would be documented within methods.
    
    Methods:
        __init__:
             Constructor to initialize an LLMContext instance with respective context, execution context, and budget parameters.
        from_context:
             Static method to create a new LLMContext from an existing ContextBase, optionally using a different budget.
        empty:
             Static method to create a new, empty LLMContext with defaults for agent context storage, execution context, and an unlimited budget.
        new_for:
             Method to generate a new LLMContext for the specified monitored object, maintaining the current context state.
        Inherits:
        ContextBase:
             A base class providing the foundational context structure for agent-based systems.

    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget) -> None:
        """
        Initialize a new instance of the class with the provided context store, execution context, and budget.
        This constructor initializes the superclass with the given arguments, setting up the necessary context for operations.
        
        Args:
            store (AgentContextStore):
                 The context store that holds the agent's state and interaction contexts.
            execution_context (ExecutionContext):
                 The context within which the current execution will operate. It can hold information like current user, permissions, and other operational details.
            budget (Budget):
                 An object representing the budget allocated for the agent's tasks or operations.
            

        """
        super().__init__(store, execution_context, budget)

    @staticmethod
    def from_context(context: ContextBase, monitored: Monitored, budget: Optional[Budget] = None) -> LLMContext:
        """
        Creates a new LLMContext instance as a copy of an existing context, with a modified execution context for a provided `monitored` object.
        This static method constructs a new `LLMContext` object using the same `AgentContextStore` from the original context but generates a new `ExecutionContext` tailored for the `monitored` parameter. An optional `Budget` can be provided to overwrite the budget of the new context, otherwise, it defaults to the budget of the original context.
        
        Parameters:
            context (ContextBase):
                 The original context to copy from.
            monitored (Monitored):
                 The object that the new execution context will be monitoring.
            budget (Optional[Budget]):
                 Optionally, a new budget for the new context. If not provided, the budget is taken from the original context.
        
        Returns:
            (LLMContext):
                 A new instance of `LLMContext` with the updated parameters for `monitored` and optionally `budget`.
            

        """
        return LLMContext(context._store, context._execution_context.new_for(monitored), budget or context._budget)

    @staticmethod
    def empty() -> LLMContext:
        """
        Creates a new instance of `LLMContext` with default, empty components.
        This static method returns an `LLMContext` object initialized with a new `AgentContextStore` containing an empty `ChatHistory`,
        a fresh `ExecutionContext`, and an `InfiniteBudget`. It is typically used to generate a clean context state.
        
        Returns:
            (LLMContext):
                 A new instance of `LLMContext` with default, uninitialized context components.
            

        """
        return LLMContext(AgentContextStore(ChatHistory()), ExecutionContext(), InfiniteBudget())

    def new_for(self, monitored: Monitored) -> LLMContext:
        """
        Creates a new LLMContext instance for a given monitored object.
        
        Args:
            monitored (Monitored):
                 The object that will be monitored within the new context.
        
        Returns:
            (LLMContext):
                 A new context object created from the current context with the given monitored instance.
            

        """
        return self.from_context(self, monitored)
