"""

A module providing a centralized context store for an agent's execution context and history during an interaction session.

This module consists of the `AgentContextStore` class, which encapsulates all components related to the state and
progress of an agent's iterations, execution log, chat history, and cancellation token. The context store enables tracking and
managing the flow of a conversation over multiple iterations, evaluation of chat messages, and the ability to handle
interruption requests through the cancellation token.

Classes:
    AgentContextStore: Holds the state and management facilities for a single agent's conversation context over
    multiple iterations.

This module also manages the importation of necessary classes such as `Iterable`, `List`, `Sequence`,
`AgentIterationContextStore`, `CancellationToken`, `ChatHistory`, `ScoredChatMessage`, `ExecutionLog`,
`MessageCollection`, and `MessageList`.


"""
from typing import Iterable, List, Sequence

from ._agent_iteration_context_store import AgentIterationContextStore
from ._cancellation_token import CancellationToken
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._execution_log import ExecutionLog
from ._message_collection import MessageCollection
from ._message_list import MessageList


class AgentContextStore:
    """
    A data structure designed to track and manage the context of an agent's interactions over the course of multiple iterations or exchanges within a conversation or series of tasks.
    
    Attributes:
        _cancellation_token (CancellationToken):
             A token to manage cancellation signals across conversations or tasks.
        _chat_history (ChatHistory):
             An object that stores the history of the chat or conversation.
        _iterations (List[AgentIterationContextStore]):
             A list of iterations, where each iteration represents a single context of the agent's interaction.
        _log (ExecutionLog):
             An object that records the execution log data throughout the agent's interactions.
    
    Methods:
        cancellation_token:
             Returns the CancellationToken associated with the agent context.
        chat_history:
             Returns the ChatHistory instance containing the record of the conversation.
        iterations:
             Returns a sequence of AgentIterationContextStore instances representing each iteration of context.
        current_iteration:
             Returns the most recent AgentIterationContextStore instance in the iterations list.
        execution_log:
             Returns the ExecutionLog instance containing log information.
        new_iteration:
             Initializes a new AgentIterationContextStore and appends it to the list of iterations.
        chain_iterations:
             Yields a sequence of MessageCollection objects corresponding to a specific chain name across all iterations.
        evaluation_history:
             Yields a sequence of scored chat messages from the evaluator within each iteration.

    """

    def __init__(self, chat_history: ChatHistory):
        """
        Initializes the context for managing chat history and iterations within an agent environment.
        
        Args:
            chat_history (ChatHistory):
                 The chat history object that stores past interactions.
        
        Attributes:
            _cancellation_token (CancellationToken):
                 Token to signal cancellation of ongoing operations.
            _chat_history (ChatHistory):
                 Stores the conversation history between the agent and the environment.
            _iterations (List[AgentIterationContextStore]):
                 A list to store context information for each iteration of the agent's operation.
            _log (ExecutionLog):
                 Logs execution details that occur within the agent's iterations.

        """
        self._cancellation_token = CancellationToken()
        self._chat_history = chat_history
        self._iterations: List[AgentIterationContextStore] = []
        self._log = ExecutionLog()

    @property
    def cancellation_token(self) -> CancellationToken:
        """
        Gets the cancellation token associated with this object.
        This property returns the `CancellationToken` instance that can be used to
        observe and respond to the cancellation requests.
        
        Returns:
            (CancellationToken):
                 The cancellation token for this object.

        """
        return self._cancellation_token

    @property
    def chat_history(self) -> ChatHistory:
        """
        Gets the chat history maintained by this instance.
        This property method returns the _chat_history attribute which presumably stores the
        chat history in a ChatHistory object. The ChatHistory class is not defined within this
        snippet, but it should encapsulate functionality and data structures to handle the chat history.
        
        Returns:
            (ChatHistory):
                 An object that represents the chat history.

        """
        return self._chat_history

    @property
    def iterations(self) -> Sequence[AgentIterationContextStore]:
        """
        Gets the iterations of the AgentIterationContextStore sequence.
        This property is used to retrieve the current sequence of iterations stored within
        the AgentIterationContextStore. Access to this property is read-only.
        
        Returns:
            (Sequence[AgentIterationContextStore]):
                 A read-only sequence of AgentIterationContextStore objects, representing the iterations.

        """
        return self._iterations

    @property
    def current_iteration(self) -> AgentIterationContextStore:
        """
        Gets the current iteration context of the agent.
        This property returns the last item in the iterations list, which contains
        contextual information about the agent's current iteration state.
        The returned object is expected to be of the type AgentIterationContextStore.
        
        Returns:
            (AgentIterationContextStore):
                 An object encapsulating the current iteration context for the agent.
        
        Raises:
            IndexError:
                 If trying to access the current iteration when the iterations list is empty.

        """
        return self._iterations[-1]

    @property
    def execution_log(self) -> ExecutionLog:
        """
        Gets the execution log of the current context.
        
        Returns:
            (ExecutionLog):
                 An object representing the execution log.

        """
        return self._log

    def new_iteration(self) -> None:
        """
        Generates a new iteration context store and appends it to the iterations list.
        This method is intended to be called when a new iteration cycle begins, ensuring a fresh collection
        of message chains and evaluation data is created for the Agent. The new iteration context store is
        instantiated and then appended to the internal list that tracks all iteration contexts for the Agent.
        
        Note that this method does not return any value.
        
        Raises:
            This method makes no assumptions about possible errors or exceptions, and as such, does not
            explicitly raise any exceptions as part of its documented interface. However, any exceptions
            that arise from underlying data structures or from the initialization of AgentIterationContextStore
            will propagate up the call stack if not handled within this method.

        """
        iteration = AgentIterationContextStore()
        self._iterations.append(iteration)

    def chain_iterations(self, name: str) -> Iterable[MessageCollection]:
        """
        Describes a function within a class that yields chained iterations of `MessageCollection` objects.
        This function takes a `name` argument which is a string identifier for the iterations
        to be chained. It yields an `Iterable` of `MessageCollection` instances from the
        underlying iterations based on the provided `name`. If the iteration does not
        contain a chain with the specified name, it yields a default `MessageList` object.
        
        Args:
            name (str):
                 The identifier for the chain iterations to yield.
        
        Yields:
            Iterable[MessageCollection]:
                 An iterator over the chained `MessageCollection`
                instances corresponding to the specified name.
            

        """
        default = MessageList()
        for iteration in self._iterations:
            yield iteration.chains.get(name, default)

    @property
    def evaluation_history(self) -> Iterable[Sequence[ScoredChatMessage]]:
        """
        
        Returns an iterable sequence of `ScoredChatMessage` objects representing the evaluation history.
            This property method yields the evaluator object from each iteration in the object's evaluation history. Each evaluator object typically includes information such as the score or feedback provided for a set of chat messages.
        
        Returns:
            (Iterable[Sequence[ScoredChatMessage]]):
                 An iterable of sequences that contains the `ScoredChatMessage` objects.

        """
        for iteration in self._iterations:
            yield iteration.evaluator
