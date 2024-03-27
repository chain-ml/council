"""

Module for managing agent context within a conversational AI framework.

This module provides the AgentContext class which acts as a container for all the context-related information required for handling and processing a conversation within an AI system. It encapsulates the chat history, execution context, budgeting for resources, and provides a structure for managing iterations of the conversation and the messages exchanged between users and the AI.

The main features of the AgentContext class include:
- Creating an empty context with default or specified budget.
- Building context from an existing chat history.
- Starting a new iteration (turn) in a conversation.
- Generating new agent contexts for different monitored components or execution units within the conversation.
- Accessing and manipulating message chains and evaluation results associated with the current iteration.
- It extends the capabilities provided by the ContextBase class.

Typical usage of this class would involve creating an agent context at the start of a conversation and then updating it as the conversation progresses with additional messages, iterations, and evaluations of responses.


"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._message_collection import MessageCollection
from ._monitored import Monitored


class AgentContext(ContextBase):
    """
    A context manager class for agents that extends ContextBase.
    AgentContext is responsible for managing the context of an agent as it interacts with the environment. It provides
    facilities for initializing the context, creating new contexts for separate iterations or execution units,
    and for managing the evaluation of messages within a conversation.
    
    Attributes:
        _store (AgentContextStore):
             An instance that encapsulates the state and history relevant to the agent's context.
        _execution_context (ExecutionContext):
             An instance that contains execution-related information such as logs.
        _budget (Budget):
             An instance representing the budget constraints for the agent's operations.
    
    Methods:
        __init__:
             Initializes a new instance of AgentContext with the provided store, execution context, and budget.
        empty:
             Creates an empty AgentContext with an optional budget.
        from_chat_history:
             Constructs an AgentContext from a ChatHistory instance and an optional budget.
        from_user_message:
             Constructs an AgentContext from a user message and an optional budget.
        new_agent_context_for:
             Creates a new AgentContext for a monitored entity.
        new_iteration:
             Starts a new iteration within the agent's context.
        new_agent_context_for_new_iteration:
             Creates a new AgentContext for the next iteration.
        new_agent_context_for_execution_unit:
             Creates a new AgentContext for the given execution unit.
        chains:
             Returns the current iterable of MessageCollection from the agent's store.
        evaluation:
             Returns the current evaluation sequence consisting of ScoredChatMessages.
        set_evaluation:
             Updates the evaluation with an iterable of ScoredChatMessages.
            The class contains both instance properties and methods, as well as static methods that provide alternative ways to
            create instances of the context.

    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget) -> None:
        """
        Initializes a new instance of the class.
        This constructor initializes the class with the given AgentContextStore, ExecutionContext, and Budget. It invokes
        the constructor of the superclass as part of the initialization process.
        
        Args:
            store (AgentContextStore):
                 An instance of AgentContextStore representing the storage of agent context information.
            execution_context (ExecutionContext):
                 An instance of ExecutionContext which provides context for execution of
                tasks such as current user details, permissions, and other relevant data.
            budget (Budget):
                 An instance of Budget representing the financial constraints or budget available for
                the execution of tasks.
            

        """
        super().__init__(store, execution_context, budget)

    @staticmethod
    def empty(budget: Optional[Budget] = None) -> AgentContext:
        """
        
        Returns a new instance of AgentContext with an empty chat history and an optional budget.
            This static method creates an AgentContext object initializing its chat history with an empty ChatHistory.
        
        Args:
            budget (Optional[Budget]):
                 An optional budget for the agent. If provided, it sets the
                budget attribute in the AgentContext instance. If not provided, the default is None,
                indicating no budget constraints.
        
        Returns:
            (AgentContext):
                 A new instance of AgentContext with its chat history initialized to an
                empty ChatHistory and its budget set as provided by the argument (if any).

        """
        return AgentContext.from_chat_history(ChatHistory(), budget)

    @staticmethod
    def from_chat_history(chat_history: ChatHistory, budget: Optional[Budget] = None) -> AgentContext:
        """
        Creates a new instance of `AgentContext` using the provided `ChatHistory` and an optional `Budget`.
        
        Args:
            chat_history (ChatHistory):
                 The chat history to base the `AgentContext` on.
            budget (Optional[Budget], optional):
                 The budget for the agent's operation. If not provided, the default budget will be used.
        
        Returns:
            (AgentContext):
                 A new instance of `AgentContext` initialized with the provided chat history and budget.

        """
        store = AgentContextStore(chat_history)
        return AgentContext(store, ExecutionContext(store.execution_log, "agent"), budget or Budget.default())

    @staticmethod
    def from_user_message(message: str, budget: Optional[Budget] = None) -> AgentContext:
        """
        Converts a user message into an AgentContext object, optionally considering a budget.
        This static method takes a user message in the form of a string and optionally a Budget
        object. It processes the user message to create a ChatHistory object, which is then used
        to instantiate an AgentContext. The AgentContext object encapsulates the current state
        of an agent's understanding of the user interaction which can include conversational context,
        extracted entities, intent recognition, and other relevant metadata.
        
        Args:
            message (str):
                 The message from the user that will be processed to create an AgentContext.
            budget (Optional[Budget]):
                 An optional Budget object that may influence the creation
                of the AgentContext, depending on the financial constraints or allocations associated
                with the interaction. If not provided, the AgentContext is created without budget constraints.
        
        Returns:
            (AgentContext):
                 An instance of AgentContext generated from the given user message
                and the optional budget.
        
        Note:
            This constructor-like method represents a simplified interface for converting a
            user's text message directly into a context object that a conversational agent can
            use in subsequent processing steps.

        """
        return AgentContext.from_chat_history(ChatHistory.from_user_message(message), budget)

    def new_agent_context_for(self, monitored: Monitored) -> AgentContext:
        """
        Creates a new AgentContext instance for a given monitored entity.
        The method takes a `monitored` object and creates a new AgentContext that is a duplicate of the
        current context but with an updated execution context specific to the monitored entity.
        
        Args:
            monitored (Monitored):
                 The entity to be monitored within the new context.
        
        Returns:
            (AgentContext):
                 A new instance of AgentContext tailored for the monitored entity.

        """
        return AgentContext(self._store, self._execution_context.new_for(monitored), self._budget)

    def new_iteration(self) -> None:
        """
        Performs the creation of a new iteration within the internal '_store' attribute of the object.
        This method acts as a wrapper to call the 'new_iteration' method of the '_store' object. It is used to signify
        a fresh cycle or iteration in whatever context '_store' is being used (e.g., a new loop, a reset of state,
        etc.). This is a management function and does not return any value.
        
        Returns:
            None

        """
        self._store.new_iteration()

    def new_agent_context_for_new_iteration(self) -> AgentContext:
        """
        Creates a new `AgentContext` instance representing a new iteration of the current context.
        This method increments the iteration count for the current agent context, generates a unique name indicating the new iteration index, and returns a new `AgentContext` instance with the updated execution context and the same budget.
        
        Returns:
            (AgentContext):
                 A new instance of `AgentContext` for the next iteration.
            

        """
        self.new_iteration()
        name = f"iterations[{len(self._store.iterations) - 1}]"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    def new_agent_context_for_execution_unit(self, name: str) -> AgentContext:
        """
        Creates a new AgentContext object for a specific execution unit with a modified name. This method is used to generate a new context for the agent that is tied to a particular unit of execution within the system, allowing for the tracing and management of different execution flows within the agent's operation.
        
        Args:
            name (str):
                 A string that uniquely identifies the execution unit within the agent's lifecycle.
        
        Returns:
            (AgentContext):
                 A new instance of AgentContext with an updated execution context to reflect the specific execution unit.

        """
        name = f"execution({name})"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    @property
    def chains(self) -> Iterable[MessageCollection]:
        """
        
        Returns an iterable of MessageCollection objects from the current iteration's chains.
            This property provides access to values of the `chains` dictionary from the current iteration
            stored in `_store`. Each value in this dictionary is a MessageCollection object, which
            holds messages or other related data.
        
        Returns:
            (Iterable[MessageCollection]):
                 An iterable collection of MessageCollection instances
                representing the chains of messages for the current iteration.

        """
        return self._store.current_iteration.chains.values()

    @property
    def evaluation(self) -> Sequence[ScoredChatMessage]:
        """
        Retrieves the current evaluation scores for chat messages.
        This property acts as a gateway to access the evaluator from the current iteration
        of the stored data. The evaluator is responsible for scoring chat messages based
        on a certain set of criteria.
        
        Returns:
            (Sequence[ScoredChatMessage]):
                 A sequence of scored chat messages, which are
                the chat messages coupled with their respective scores according
                to the evaluation criteria.

        """
        return self._store.current_iteration.evaluator

    def set_evaluation(self, messages: Iterable[ScoredChatMessage]) -> None:
        """
        Sets the evaluation scores for chat messages in the current iteration.
        
        Args:
            messages (Iterable[ScoredChatMessage]):
                 An iterable of ScoredChatMessage objects, each containing a chat message and its associated score.
                This method updates the current iteration's evaluator with the new set of messages and their scores. The actual storing mechanism is abstracted within the '_store' attribute's current_iteration object.
                No return value as the method is meant to update internal state.

        """
        self._store.current_iteration.set_evaluator(messages)
