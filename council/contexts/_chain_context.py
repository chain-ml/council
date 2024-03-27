"""

Module: _chain_context

This module defines the ChainContext class, which is a specialized context for managing the message chain
within an agent interaction. The ChainContext class provides functionality for working with a series
of chat messages within the context of a single chain, taking into account the agent's execution history,
budget constraints, and cancellation tokens.

A chain in this context refers to a sequence of messages that form a conversational thread between
an agent and a user, possibly involving various skills and external services. A chain context is intended
to offer utility methods for accessing and manipulating the messages comprising the chain, conducting checks
related to the chain's execution (such as budget expiration or cancellation conditions), and forking
or merging chain contexts for parallel processing.

In addition to the primary ChainContext class, the module also includes the necessary type imports that
support the functionality of the ChainContext, as well as any relevant abstract base classes or generics.

Classes:
    ChainContext - Extends ContextBase and implements MessageCollection to represent a single message chain
                   in an agent's context, handling message operations and context management functionality
                   within an ongoing agent-user interaction.

Functions:
    ChainContext.from_agent_context - Static method to create a ChainContext from an existing AgentContext,
                                     a monitored object, and an optional budget.
    ChainContext.from_chat_history - Static method to create a ChainContext directly from a given ChatHistory
                                     and an optional budget.
    ChainContext.from_user_message - Static method to create a ChainContext from a single user message and an
                                     optional budget.
    ChainContext.empty - Static method to create an empty ChainContext with no messages and the default budget.

The module uses abc (Abstract Base Class) module and other internal modules (e.g., AgentContext, ExecutionContext)
to provide the necessary functionality for the ChainContext. It also includes the more_itertools library for
advanced iteration tools.


"""
from __future__ import annotations

from typing import Iterable, List, Optional

import more_itertools

from ._agent_context import AgentContext
from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._cancellation_token import CancellationToken
from ._chat_history import ChatHistory
from ._chat_message import ChatMessage
from ._composite_message_collection import CompositeMessageCollection
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._message_collection import MessageCollection
from ._message_list import MessageList
from ._monitored import Monitored


class ChainContext(ContextBase, MessageCollection):
    """
    A container to handle context for chains of conversations, inheriting from ContextBase and MessageCollection.
    ChainContext provides functionality to manage and interact with chains of conversation messages. It keeps track of the current and previous messages, allowing for the handling of multiple iterations of conversations. It also offers the ability to fork the context, merge contexts, and manipulate messages within a chain.
    
    Attributes:
        _name (str):
             The name identifying the chain.
        _current_messages (MessageList):
             A collection of messages for the current iteration.
        _previous_messages (MessageList):
             A collection of messages from the previous iteration.
        _current_iteration_messages (CompositeMessageCollection):
             Combined messages of previous and current iterations for the current chain context.
        _previous_iteration_messages (CompositeMessageCollection):
             Combined messages of all but the last iteration from the chain.
        _all_iteration_messages (CompositeMessageCollection):
             A collection of all messages across iterations.
        _all_messages (CompositeMessageCollection):
             A collection of all messages, including the chat history and all iterations.
    
    Methods:
        __init__:
             Initializes a new instance of the ChainContext class.
        from_agent_context:
             Creates a new ChainContext instance from an existing AgentContext.
        fork_for:
             Creates a new ChainContext as a fork for a specific monitored object with an optional budget.
        should_stop:
             Checks if the context should stop processing based on budget expiration or cancellation token.
        merge:
             Merges the current messages from other ChainContext instances into this one.
        append:
             Appends a single ChatMessage to the current message collection.
        extend:
             Appends multiple ChatMessages to the current message collection.
        from_chat_history:
             Creates a new ChainContext instance from a given ChatHistory.
        from_user_message:
             Creates a new ChainContext from a user's message string.
        empty:
             Creates a new, empty ChainContext instance.
        Properties:
        cancellation_token:
             Returns the CancellationToken associated with the context.
        budget:
             Returns the budget assigned to the context.
        messages:
             Returns an iterable of all ChatMessages in the context.
        reversed:
             Returns a reverse iterable of ChatMessages.
        chain_histories:
             Yields all chain histories from the iterations.
        current:
             Returns the MessageCollection for the current chain iteration.

    """

    def __init__(
        self,
        store: AgentContextStore,
        execution_context: ExecutionContext,
        name: str,
        budget: Budget,
        messages: Optional[Iterable[ChatMessage]] = None,
    ):
        """
        Initializes a new instance with the specified context and message processing capabilities.
        This constructor prepares the internal messaging infrastructure by organizing messages
        from current and previous iterations and the overall chat history into various collections.
        The constructed instance has access to all messages from the current iteration,
        previous iterations, as well as the entire conversation history.
        
        Args:
            store (AgentContextStore):
                 The storage facility for agent contexts.
            execution_context (ExecutionContext):
                 The execution context for this instance.
            name (str):
                 The name identifier for this instance.
            budget (Budget):
                 The budget allocated for this instance's operation.
            messages (Optional[Iterable[ChatMessage]]):
                 An optional iterable of ChatMessage instances
                that should be included as part of the current messages. Defaults to None.
        
        Raises:
            Whatever exceptions can be raised by the superclass's __init__ method or by
            the process of accessing the message collections from the provided `store`.
            

        """
        super().__init__(store, execution_context, budget)
        self._name = name
        self._current_messages = MessageList()
        self._previous_messages = MessageList(messages)

        self._current_iteration_messages = CompositeMessageCollection([self._previous_messages, self._current_messages])
        self._previous_iteration_messages = CompositeMessageCollection(
            list(self._store.chain_iterations(self._name))[:-1]
        )
        self._all_iteration_messages = CompositeMessageCollection(
            [self._previous_iteration_messages, self._current_iteration_messages]
        )
        self._all_messages = CompositeMessageCollection([self.chat_history, self._all_iteration_messages])

    @property
    def cancellation_token(self) -> CancellationToken:
        """
        
        Returns the CancellationToken associated with this instance.
            This property provides access to the CancellationToken that can be used to handle cancellation requests in asynchronous and synchronous operations associated with this instance.
        
        Returns:
            (CancellationToken):
                 An instance of CancellationToken that signals cancellation.

        """
        return self._store.cancellation_token

    @property
    def budget(self) -> Budget:
        """
        Gets the budget associated with an instance.

        """
        return self._budget

    @property
    def messages(self) -> Iterable[ChatMessage]:
        """
        Property that retrieves all chat messages from a collection.
        This property acts as a getter that provides access to an iterable collection
        of ChatMessage objects, which represent individual messages within a chat.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable containing ChatMessage instances,
                allowing the user to iterate over the collection of messages.
            

        """
        return self._all_messages.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        """
        
        Returns an iterable of ChatMessage objects in reversed order from the internal messages store.
        
        Returns:
            (Iterable[ChatMessage]):
                 An iterable that provides access to the stored chat messages in reverse
                order, such that the last message is returned first.

        """
        return self._all_messages.reversed

    @property
    def chain_histories(self) -> Iterable[MessageCollection]:
        """
        Retrieves an iterable of MessageCollection objects that represent the
        chain histories associated with a particular name within the iterations
        stored in an internal datastore.
        This property method iterates through the datastore's iterations,
        accesses the 'chains' attribute of each iteration, and yields the
        chain corresponding to the instance's '_name' attribute if it is present.
        
        Returns:
            (Iterable[MessageCollection]):
                 An iterable of MessageCollection instances, each
                representing a chain of messages from the datastore iterations.
            

        """
        for item in self._store.iterations:
            chain = item.chains.get(self._name)
            if chain is not None:
                yield chain

    @property
    def current(self) -> MessageCollection:
        """
        
        Returns the collection of messages for the current iteration.
        
        Returns:
            (MessageCollection):
                 An object containing messages from the current iteration of the process.

        """
        return self._current_iteration_messages

    @staticmethod
    def from_agent_context(context: AgentContext, monitored: Monitored, name: str, budget: Optional[Budget] = None):
        """
        Creates a new instance of ChainContext using an existing AgentContext and additional parameters.
        This static method initializes a ChainContext by ensuring a message chain with the given name exists within the current
        iteration of the AgentContext's store. If the specified budget is None, the default Budget is used.
        
        Args:
            context (AgentContext):
                 The progenitor agent context with the originally shared state.
            monitored (Monitored):
                 The object being monitored in the new execution context.
            name (str):
                 The name of the chain within the AgentContext's iteration to ensure or create.
            budget (Optional[Budget]):
                 The budget for the new ChainContext; defaults to Budget.default() if not provided.
        
        Returns:
            (ChainContext):
                 A newly created ChainContext object initialized with the provided AgentContext's store, a new
                execution context for the specified monitored object, the given name, and the specified or default budget.
            

        """
        context._store.current_iteration.ensure_chain_exists(name)
        return ChainContext(
            context._store, context._execution_context.new_for(monitored), name, budget or Budget.default()
        )

    def fork_for(self, monitored: Monitored, budget: Optional[Budget] = None) -> ChainContext:
        """
        Creates a new ChainContext with the provided monitored object and an optional budget. This method essentially clones the current context, but with a new execution context tailored for the specified monitored object. If no budget is provided, it inherits the budget from the current ChainContext.
        
        Args:
            monitored (Monitored):
                 The object being monitored in the new ChainContext.
            budget (Optional[Budget]):
                 An optional budget for the new ChainContext. If no budget is specified, the existing budget from the current context is used.
        
        Returns:
            (ChainContext):
                 A new instance of ChainContext with the same store, messages and name as the current context but with a new execution context for the monitored object and the specified (or inherited) budget.

        """
        return ChainContext(
            self._store,
            self._execution_context.new_for(monitored),
            self._name,
            budget or self._budget,
            more_itertools.flatten([self._previous_messages.messages, self._current_messages.messages]),
        )

    def should_stop(self) -> bool:
        """
        Determines whether the process should stop based on the status of the budget and cancellation token.
        
        Returns:
            (bool):
                 True if either the budget has expired or the cancellation token has been cancelled, otherwise False.

        """
        if self._budget.is_expired():
            self.logger.debug('message="stopping" reason="budget expired"')
            return True
        if self.cancellation_token.cancelled:
            self.logger.debug('message="stopping" reason="cancellation token is set"')
            return True

        return False

    def merge(self, contexts: List["ChainContext"]) -> None:
        """
        Merges the current message chains with the message chains in the given contexts.
        This method iterates over a list of ChainContext objects, incorporating each
        context's messages into the current ChainContext's message set. This is done by
        adding all messages from each context's _current_messages to the
        _current_messages attribute of the current context.
        
        Args:
            contexts (List['ChainContext']):
                 A list of ChainContext instances whose
                messages are to be merged into the current context.
        
        Raises:
            None:
                 This method is not expected to raise any exceptions under
                normal circumstances.

        """
        for context in contexts:
            self._current_messages.add_messages(context._current_messages.messages)

    def append(self, message: ChatMessage) -> None:
        """
        Appends a given ChatMessage to the current messages and updates the storage iteration chain.
        This method takes a ChatMessage instance and adds it to the set of current messages. Additionally,
        it appends the message to the chain in the store corresponding to the current iteration,
        using the assigned name and execution context entry.
        
        Args:
            message (ChatMessage):
                 The chat message instance to be appended.
        
        Returns:
            None

        """
        self._current_messages.add_message(message)
        self._store.current_iteration.append_to_chain(self._name, message, self._execution_context.entry)

    def extend(self, messages: Iterable[ChatMessage]) -> None:
        """
        Extends the current collection of chat messages with the given iterable of ChatMessage objects.
        This method iterates over the provided messages and appends each to the end of the current message collection. This is typically used to add a batch of new messages all at once.
        
        Args:
            messages (Iterable[ChatMessage]):
                 An iterable of ChatMessage instances to be added to the current collection.
        
        Raises:
            TypeError:
                 If the provided messages are not an iterable of ChatMessage instances, a TypeError may be raised.
        
        Returns:
            (None):
                 This method does not return anything as it modifies the collection of messages in-place.

        """
        for message in messages:
            self.append(message)

    @staticmethod
    def from_chat_history(history: ChatHistory, budget: Optional[Budget] = None) -> ChainContext:
        """
        Creates a new ChainContext from the provided chat history.
        This static method initializes a new ChainContext using the chat history and an optional budget. It begins a new iteration
        for the context, includes a mocked monitoring, and assigns a mock chain name for identification.
        
        Args:
            history (ChatHistory):
                 An instance of ChatHistory that contains the past interactions to be included in the new context.
            budget (Optional[Budget]):
                 An optional budget instance that sets limits or guidelines for the execution.
        
        Returns:
            (ChainContext):
                 A new instance of ChainContext initialized with the given chat history and optional budget parameters.
            

        """
        from ..mocks import MockMonitored

        context = AgentContext.from_chat_history(history)
        context.new_iteration()
        return ChainContext.from_agent_context(context, MockMonitored("mock chain"), "mock chain", budget)

    @staticmethod
    def from_user_message(message: str, budget: Optional[Budget] = None) -> ChainContext:
        """
        Creates a ChainContext instance from a user message.
        This static method facilitates the creation of a ChainContext object using the provided user message. If a budget is specified,
        it will be used in the creation of the ChainContext; otherwise, the budget is considered to be 'None'.
        
        Args:
            message (str):
                 The message from the user to be processed.
            budget (Optional[Budget]):
                 An optional Budget object that specifies the computational resources available.
        
        Returns:
            (ChainContext):
                 The new ChainContext instance created from the user's message and the optional budget.
            

        """
        return ChainContext.from_chat_history(ChatHistory.from_user_message(message), budget)

    @staticmethod
    def empty() -> ChainContext:
        """
        
        Returns an empty `ChainContext` object initialized with an empty `ChatHistory`.
            This static method creates an instance of `ChainContext` that contains no previously held chat history or state. It is useful for starting a new chat sequence without any prior context.
        
        Returns:
            (ChainContext):
                 An instance of `ChainContext` with an empty `ChatHistory`.

        """
        return ChainContext.from_chat_history(ChatHistory())
