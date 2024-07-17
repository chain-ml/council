from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Monitorable, ScoredChatMessage


class FilterException(Exception):
    """
    Exception raised specifically for errors encountered during filtering process.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class FilterBase(Monitorable, ABC):
    """
    Abstract base class for an agent filter.
    """

    def __init__(self) -> None:
        super().__init__("filter")

    def execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Selects responses from the agent's context.

        Args:
            context (AgentContext): The context for selecting responses.

        Returns:
            List[ScoredChatMessage]: A list of scored agent messages representing the selected responses.

        Raises:
            None
        """
        with context:
            return self._execute(context=context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        pass
