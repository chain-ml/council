from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Monitorable, ScoredChatMessage


class FilterBase(Monitorable, ABC):
    """
    Abstract base class for an agent filter.
    """

    def __init__(self):
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
