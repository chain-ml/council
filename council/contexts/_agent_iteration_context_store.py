"""

Module `_agent_iteration_context_store` is responsible for storing and managing the context pertaining to different chains of messages during an agent's iteration. It keeps track of the ongoing discussions (chains) with different entities, such as users or other agents, and evaluates messages based on certain criteria or scores. The module provides functionality to ensure the integrity and accessibility of these conversation chains and scored messages throughout the lifecycle of an agent's processing loop.

Classes:
    - AgentIterationContextStore: A storage for maintaining the current state of message chains and the evaluation of messages.

Attributes:
    - chains (Mapping[str, MessageCollection]): A mapping of chain names to MonitoredMessageList objects that represent ongoing conversations.
    - evaluator (Sequence[ScoredChatMessage]): A list of ScoredChatMessage objects representing messages that have been evaluated and scored.

Functions:
    - set_evaluator(value: Iterable[ScoredChatMessage]) -> None: Replaces the current evaluator list with a new sequence of ScoredChatMessages.
    - ensure_chain_exists(name: str) -> None: Ensures that a MonitoredMessageList for a given chain name is created and available in the store.
    - append_to_chain(chain: str, message: ChatMessage, log_entry: ExecutionLogEntry) -> None: Appends a new ChatMessage to the specified chain and logs the action in the provided ExecutionLogEntry.


"""
from typing import Dict, Iterable, List, Mapping, Sequence

from ._chat_message import ChatMessage, ScoredChatMessage
from ._execution_log_entry import ExecutionLogEntry
from ._message_collection import MessageCollection
from ._message_list import MessageList
from ._monitored_message_list import MonitoredMessageList


class AgentIterationContextStore:
    """
    A storage mechanism for agent iteration contexts, maintaining message chains and evaluator lists.
    This class acts as a repository for storing and manipulating a collection of message chains and
    evaluators corresponding to an agent's current iterations. Message chains are stored as a mapping
    from chain names to MonitoredMessageLists, while the evaluator is a list of ScoredChatMessages used
    for evaluating agent performance.
    
    Attributes:
        _chains (Dict[str, MonitoredMessageList]):
             A private dictionary mapping chain names to
            MonitoredMessageLists which are used to monitor message chains.
        _evaluator (List[ScoredChatMessage]):
             A private list to hold ScoredChatMessages for the purpose
            of evaluating agent interactions.
    
    Methods:
        __init__(self):
            Initializes a new instance of AgentIterationContextStore, creating empty storage for message
            chains and evaluator.
        chains(self) -> Mapping[str, MessageCollection]:
            Property to access the stored message chains.
        evaluator(self) -> Sequence[ScoredChatMessage]:
            Property to access the stored evaluator messages.
        set_evaluator(self, value:
             Iterable[ScoredChatMessage]) -> None:
            Replaces the current evaluator list with messages from the given iterable.
        ensure_chain_exists(self, name:
             str) -> None:
            Ensures that a message chain exists for the given name, creating a new MonitoredMessageList
            mapped to that name if it doesn't already exist.
        append_to_chain(self, chain:
             str, message: ChatMessage, log_entry: ExecutionLogEntry) -> None:
            Appends a given message and its corresponding log entry to the specified message chain.

    """

    _chains: Dict[str, MonitoredMessageList]
    _evaluator: List[ScoredChatMessage]

    def __init__(self):
        """
        Initializes a new instance of the class with default properties.
        
        Attributes:
            self._chains (dict):
                 A private dictionary to store chain data or mappings.
            self._evaluator (list):
                 A private list to hold evaluator objects or references.

        """
        self._chains = {}
        self._evaluator = []

    @property
    def chains(self) -> Mapping[str, MessageCollection]:
        """
        
        Returns the mapping of chains associated with the current instance.
            This property method returns the private _chains attribute, which is a mapping from
            string identifiers to MessageCollection objects. Each MessageCollection may represent a sequence
            of messages or a conversation thread that the object of the current class instance manages or holds.
        
        Returns:
            (Mapping[str, MessageCollection]):
                 A dictionary-like object mapping string keys to MessageCollection instances, where the keys uniquely identify each message chain.
            

        """
        return self._chains

    @property
    def evaluator(self) -> Sequence[ScoredChatMessage]:
        """
        Property that returns the evaluator of the object.
        The evaluator is a sequence of scored chat messages. This property allows access to the internal
        evaluator state, which consists of chat messages that have been scored based on certain criteria deemed
        relevant to the evaluation process.
        
        Returns:
            (Sequence[ScoredChatMessage]):
                 A sequence of scored chat messages contained in the evaluator.
            

        """
        return self._evaluator

    def set_evaluator(self, value: Iterable[ScoredChatMessage]) -> None:
        """
        Sets the evaluator with a new set of scored chat messages.
        This method clears the existing evaluator data and populates it with the new
        set of scored chat messages provided as an argument. It is expected that
        the value is an iterable of `ScoredChatMessage` objects.
        
        Args:
            value (Iterable[ScoredChatMessage]):
                 An iterable of `ScoredChatMessage`
                objects to be used as the new set of evaluator data.
        
        Returns:
            None
            

        """
        self._evaluator.clear()
        self._evaluator.extend(value)

    def ensure_chain_exists(self, name: str) -> None:
        """
        Ensures that a messaging chain with the given name exists within the object's internal mapping.
        If a chain with the provided name does not exist, this method initializes a new MonitoredMessageList with an empty MessageList and adds it to the internal mapping under the given name.
        
        Args:
            name (str):
                 The name of the chain to ensure exists.
        
        Returns:
            None

        """
        self._chains[name] = MonitoredMessageList(MessageList())

    def append_to_chain(self, chain: str, message: ChatMessage, log_entry: ExecutionLogEntry) -> None:
        """
        Appends a message and its corresponding execution log entry to a specified chain within the object's internal chain collection.
        
        Args:
            chain (str):
                 The name of the chain to which the message and log entry will be appended. This chain should already exist in the object's chain collection.
            message (ChatMessage):
                 The message object that contains the information to be appended to the chain.
            log_entry (ExecutionLogEntry):
                 The log entry object associated with the message that provides details about the execution context.
        
        Returns:
            None
        
        Raises:
            KeyError:
                 If the specified chain does not exist in the object's internal chain collection.

        """
        self._chains[chain].append(message, log_entry)
