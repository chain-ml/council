from typing import Any, Dict, List, Optional

from .messages import ChatMessageBase, UserMessage, AgentMessage, SkillMessage, ScoredAgentMessage
from .cancellation_token import CancellationToken
from council.utils import Option


class ChatHistory:
    """
    represents the history of messages exchanged between the user and the :class:`~.Agent`

    Attributes:
        messages(list[ChatMessageBase]): list of messages
    """

    messages: List[ChatMessageBase] = []

    def __init__(self):
        """
        initialize a new instance
        """

        self.messages = []

    def add_user_message(self, message: str):
        """
        adds a :class:`UserMessage` into the history

        Arguments:
            message (str): a text message
        """

        self.messages.append(UserMessage(message))

    def last_user_message(self) -> Option[UserMessage]:
        """
        get the most recent user message in the history, if any.

        Returns:
            Option[UserMessage]: an :class:`council.utils.Option` with a user message, if any
        """
        for message in reversed(self.messages):
            if isinstance(message, UserMessage):
                return Option(message)
        return Option.none()

    def add_agent_message(self, message: str, data: Any = None):
        """
        adds a :class:`AgentMessage` into the history

        Arguments:
            message (str): a text message
            data (Any): some data, if any
        """

        self.messages.append(AgentMessage(message, data))

    def last_agent_message(self) -> Option[AgentMessage]:
        """
        get the most recent agent message in the history, if any.

        Returns:
            Option[AgentMessage]: an :class:`council.utils.Option` with an agent message, if any
        """

        for message in reversed(self.messages):
            if isinstance(message, AgentMessage):
                return Option(message)
        return Option.none()

    def last_message(self) -> Option[ChatMessageBase]:
        """
        get the most recent message in the history, if any.

        Returns:
            Option[ChatMessageBase]: an :class:`council.utils.Option` with an agent message, if any
        """

        return Option.none() if len(self.messages) == 0 else Option.some(self.messages[-1])

    @staticmethod
    def from_user_message(message: str) -> "ChatHistory":
        history = ChatHistory()
        history.add_user_message(message=message)
        return history


class ChainHistory:
    """
    Manages all the :class:`SkillMessage` generated during one execution of a :class:`.Chain`

    Attributes:
        messages (List[SkillMessage]): list of :class:`SkillMessage`
    """

    messages: List[SkillMessage]

    def __init__(self):
        """Initialize a new instance"""
        self.messages = []

    def last_message(self) -> Option[SkillMessage]:
        """
        Get the last (most recent) message, if any, added by a :class:`.SkillBase`

        Returns:
            Option[SkillMessage]: an :class:`council.utils.Option` wrapping a :class:`.SkillMessage` if any.
        """
        if len(self.messages) > 0:
            return Option(self.messages[-1])
        return Option.none()

    def last_message_from(self, skill: str) -> Option[SkillMessage]:
        """
        Get the last (most recent) message, if any, added by specific :class:`.SkillBase` identified by its name.

        Parameters:
            skill (str): the name of the skill
        Returns:
            Option[SkillMessage]: an :class:`council.utils.Option` wrapping a :class:`.SkillMessage` if any.
        """
        for message in self.messages[::-1]:
            if message.is_from_skill(skill):
                return Option(message)
        return Option.none()


class ChainContext:
    """
    Class representing the execution context of a :class:`.Chain`.

    Attributes:
        chatHistory (ChatHistory): The chat history.
        chainHistory (List[ChainHistory]): The chain iteration history for the chain.
    """

    chatHistory: ChatHistory
    chainHistory: List[ChainHistory]

    def __init__(self, chat_history: ChatHistory, chain_history: List[ChainHistory]):
        """
        Initializes the ChainContext with the provided chat and chain history.

        Args:
            chat_history (ChatHistory): The chat history.
            chain_history (List[ChainHistory]): All the :class:`ChainHistory` from the many execution of a chain.
        """
        self.chatHistory = chat_history
        self.chainHistory = chain_history
        self.cancellationToken = CancellationToken()

    def new_iteration(self):
        """
        Prepare this instance for a new execution of a chain by adding a new :class:`ChainHistory`
        """
        self.chainHistory.append(ChainHistory())

    @property
    def current(self) -> ChainHistory:
        """
        Returns the :class:`ChainHistory` to be used for the current execution of a :class:`.Chain`

        Returns:
            ChainHistory: the chain history
        """
        return self.chainHistory[-1]

    @property
    def last_message(self) -> Option[ChatMessageBase]:
        last_chain_message = self.current.last_message()
        if last_chain_message.is_none():
            return self.chatHistory.last_message()
        return Option.some(last_chain_message.unwrap())

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
        super().__init__(chain_context.chatHistory, chain_context.chainHistory)
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
