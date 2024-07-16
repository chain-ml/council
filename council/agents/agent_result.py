from collections.abc import Sequence
from typing import List, Optional

from council.contexts import ChatMessage, ScoredChatMessage
from council.utils import Option


class AgentResult:
    """
    Represent the execution result of an :class:`Agent`
    """

    def __init__(self, messages: Optional[List[ScoredChatMessage]] = None) -> None:
        """
        Initialize a new instance.

        Parameters:
            messages(Optional[List[ScoredChatMessage]]): an optional list of messages
        """
        self._messages: List[ScoredChatMessage] = messages if messages is not None else []

    @property
    def messages(self) -> Sequence[ScoredChatMessage]:
        """
        An unordered list of messages, with their scores.

        Returns:
            Sequence[ScoredChatMessage]:
        """
        return self._messages

    @property
    def best_message(self) -> ChatMessage:
        """
        The message with the highest score. If multiple messages have the highest score, the first one is returned.

        Returns:
            ChatMessage:

        Raises:
            ValueError: there is no messages
        """
        return max(self._messages, key=lambda item: item.score).message

    @property
    def try_best_message(self) -> Option[ChatMessage]:
        """
        The message with the highest score, if any. See :meth:`best_message` for more details

        Returns:
            Option[ChatMessage]: the message with the highest score, wrapped into :meth:`.Option.some`, if some,
                :meth:`.Option.none` otherwise
        """
        if len(self._messages) == 0:
            return Option.none()
        return Option.some(self.best_message)
