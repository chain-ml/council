from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, ScoredChatMessage
from council.runners import Budget


class FilterBase(ABC):
    """
    Abstract base class for an agent controller.

    """

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
        return self._execute(context=context, budget=budget)

    @abstractmethod
    def _execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        pass
