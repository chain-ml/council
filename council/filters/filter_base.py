from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Budget, ScoredChatMessage
from council.monitors import Monitorable


class FilterBase(Monitorable, ABC):
    """
    Abstract base class for an agent controller.

    """

    def __init__(self):
        super().__init__("filter")

    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        """
        Selects responses from the agent's context.

        Args:
            context (AgentContext): The context for selecting responses.
            budget (Budget): The budget for selecting responses.

        Returns:
            List[ScoredChatMessage]: A list of scored agent messages representing the selected responses.

        Raises:
            None
        """
        with context:
            return self._execute(context=context, budget=budget)

    @abstractmethod
    def _execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        pass
