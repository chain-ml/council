import abc
from typing import Any, Dict, List, Optional, Sequence, Callable, Iterable
from typing_extensions import TypeGuard

from more_itertools import first

from .messages import ChatMessageBase, UserMessage, AgentMessage, SkillMessage, ScoredAgentMessage, ChatMessageKind
from .cancellation_token import CancellationToken
from council.utils import Option


class MessageCollection(abc.ABC):
    @property
    @abc.abstractmethod
    def messages(self) -> Iterable[ChatMessageBase]:
        pass

    @property
    @abc.abstractmethod
    def reversed(self) -> Iterable[ChatMessageBase]:
        pass

    @property
    def last_message(self) -> Optional[ChatMessageBase]:
        return first(self.reversed, None)

    @property
    def try_last_message(self) -> Option[ChatMessageBase]:
        return Option(self.last_message)

    @property
    def last_user_message(self) -> Optional[ChatMessageBase]:
        return self._last_message_filter(self.message_kind_predicate(ChatMessageKind.User))

    @property
    def try_last_user_message(self) -> Option[ChatMessageBase]:
        return Option(self.last_user_message)

    @property
    def last_agent_message(self) -> Optional[ChatMessageBase]:
        return self._last_message_filter(self.message_kind_predicate(ChatMessageKind.Agent))

    @property
    def try_last_agent_message(self) -> Option[ChatMessageBase]:
        return Option(self.last_agent_message)

    def last_message_from_skill(self, skill_name: str) -> Optional[ChatMessageBase]:
        def predicate(message: ChatMessageBase):
            return (
                message.is_of_kind(ChatMessageKind.Skill)
                and isinstance(message, SkillMessage)
                and message.is_from_skill(skill_name)
            )

        return self._last_message_filter(predicate)

    def _last_message_filter(self, predicate: Callable[[ChatMessageBase], bool]) -> Optional[ChatMessageBase]:
        def typeguard_predicate(message: ChatMessageBase) -> TypeGuard[Optional[ChatMessageBase]]:
            return isinstance(message, ChatMessageBase) and predicate(message)

        return first(filter(typeguard_predicate, self.reversed), None)

    @staticmethod
    def message_kind_predicate(kind: ChatMessageKind) -> Callable[[ChatMessageBase], bool]:
        return lambda m: m.is_of_kind(kind)


class ChatHistory(MessageCollection):
    """
    represents the history of messages exchanged between the user and the :class:`.Agent`
    """

    _messages: List[ChatMessageBase] = []

    def __init__(self):
        """
        initialize a new instance
        """

        self._messages = []

    @property
    def messages(self) -> Iterable[ChatMessageBase]:
        return self._messages

    @property
    def reversed(self) -> Iterable[ChatMessageBase]:
        return reversed(self._messages)

    def add_user_message(self, message: str):
        """
        adds a :class:`UserMessage` into the history

        Arguments:
            message (str): a text message
        """

        self._messages.append(UserMessage(message))

    def add_agent_message(self, message: str, data: Any = None):
        """
        adds a :class:`AgentMessage` into the history

        Arguments:
            message (str): a text message
            data (Any): some data, if any
        """

        self._messages.append(AgentMessage(message, data))

    @staticmethod
    def from_user_message(message: str) -> "ChatHistory":
        history = ChatHistory()
        history.add_user_message(message=message)
        return history


class ChainHistory(MessageCollection):
    """
    Manages all the :class:`SkillMessage` generated during one execution of a :class:`.Chain`
    """

    _messages: List[SkillMessage]

    def __init__(self):
        """Initialize a new instance"""
        self._messages = []

    @property
    def messages(self) -> Sequence[SkillMessage]:
        return self._messages

    @property
    def reversed(self) -> Iterable[ChatMessageBase]:
        return reversed(self._messages)

    def append(self, message: SkillMessage):
        self._messages.append(message)


class ChainContext(MessageCollection):
    """
    Class representing the execution context of a :class:`.Chain`.
    """

    def __init__(self, chat_history: ChatHistory, chain_history: List[ChainHistory]):
        """
        Initializes the ChainContext with the provided chat and chain history.

        Args:
            chat_history (ChatHistory): The chat history.
            chain_history (List[ChainHistory]): All the :class:`ChainHistory` from the many execution of a chain.
        """
        self._chat_history = chat_history
        self._chain_histories = chain_history
        self._cancellation_token = CancellationToken()

    def new_iteration(self):
        """
        Prepare this instance for a new execution of a chain by adding a new :class:`ChainHistory`
        """
        self._chain_histories.append(ChainHistory())

    @property
    def cancellationToken(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def messages(self) -> Iterable[ChatMessageBase]:
        for inner_list in [self._chat_history, *self._chain_histories]:
            for item in inner_list.messages:
                yield item

    @property
    def reversed(self) -> Iterable[ChatMessageBase]:
        for inner_list in reversed([self._chat_history, *self._chain_histories]):
            for item in inner_list.reversed:
                yield item

    @property
    def chain_histories(self) -> Sequence[ChainHistory]:
        return self._chain_histories

    @property
    def chain_history(self) -> Sequence[ChainHistory]:
        return self.chain_histories

    @property
    def current(self) -> ChainHistory:
        """
        Returns the :class:`ChainHistory` to be used for the current execution of a :class:`.Chain`

        Returns:
            ChainHistory: the chain history
        """
        return self._chain_histories[-1]

    @property
    def chat_history(self) -> ChatHistory:
        return self._chat_history

    @property
    def chatHistory(self) -> ChatHistory:
        return self.chat_history

    @staticmethod
    def empty() -> "ChainContext":
        history = ChatHistory()
        return ChainContext(history, [])

    @staticmethod
    def from_user_message(message: str) -> "ChainContext":
        history = ChatHistory.from_user_message(message)
        return ChainContext(history, [])


class IterationContext:
    """
    Provides context information when running inside a loop.
    """

    def __init__(self, index: int, value: Any):
        self._index = index
        self._value = value

    @property
    def index(self) -> int:
        """
        Returns the index of the current iteration

        Returns:
            int:
        """
        return self._index

    @property
    def value(self) -> Any:
        """
        Returns the value for the current iteration

        Returns:
            Any:
        """
        return self._value

    @staticmethod
    def empty() -> Option["IterationContext"]:
        return Option.none()

    @staticmethod
    def new(index: int, value: Any) -> Option["IterationContext"]:
        return Option.some(IterationContext(index, value))


class SkillContext(ChainContext):
    """
    Class representing the execution context of a :class:`.SkillBase`.
    """

    def __init__(self, chain_context: ChainContext, iteration: Option[IterationContext]):
        super().__init__(chain_context.chatHistory, chain_context._chain_histories)
        self._iteration = iteration

    @property
    def iteration(self) -> Option[IterationContext]:
        """
        The iteration context, if any.

        Returns:
            Option[IterationContext]: Some iteration context, if any, else :meth:`.Option.none`
        """
        return self._iteration


class AgentContext:
    """
    Class representing the execution context of an :class:`.Agent`.

    Attributes:
        chatHistory (ChatHistory): The chat history.
        chainHistory (Dict[str, List[ChainHistory]]): The chain history for each :class:`.Chain`.
        evaluationHistory (List[List[ScoredAgentMessage]]): The iteration history of evaluated agent messages.
    """

    chatHistory: ChatHistory
    chainHistory: Dict[str, List[ChainHistory]]
    evaluationHistory: List[List[ScoredAgentMessage]]

    def __init__(self, chat_history: ChatHistory):
        """
        Initializes the AgentContext with the provided chat history.

        Args:
            chat_history (ChatHistory): The chat history.
        """
        self.chatHistory = chat_history
        self.chainHistory: Dict[str, List[ChainHistory]] = {}
        self.evaluationHistory = []

    def new_chain_context(self, name: str) -> ChainContext:
        """
        Creates a new chain context for the specified chain name.

        Args:
            name (str): The name of the chain.

        Returns:
            ChainContext: The new chain context.
        """
        history = self.chainHistory.get(name)
        if history is None:
            history = []
            self.chainHistory[name] = history
        history.append(ChainHistory())
        return ChainContext(self.chatHistory, history)

    def last_evaluator_iteration(self) -> Optional[List[ScoredAgentMessage]]:
        """
        Retrieves the last iteration of the evaluator's history.

        Returns:
            Optional[List[ScoredAgentMessage]]: The last iteration of the evaluator's history,
                or None if the history is empty.
        """
        if len(self.evaluationHistory) == 0:
            return None
        return self.evaluationHistory[-1]

    def last_chain_history_iteration(self, chain_name: str) -> Optional[ChainHistory]:
        """
        Retrieves the last iteration of the specified chain's history.

        Args:
            chain_name (str): The name of the chain.

        Returns:
            Optional[ChainHistory]: The last iteration of the specified chain's history,
                or None if the history is empty.

        Raises:
            None
        """
        iterations: List[ChainHistory] = self.chainHistory[chain_name]
        if iterations is None or len(iterations) == 0:
            return None
        return iterations[-1]
