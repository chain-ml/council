from abc import ABC, abstractmethod
from typing import List

from council.core.execution_context import (
    AgentContext,
    ScoredAgentMessage,
)
from council.core.budget import Budget


class EvaluatorBase(ABC):
    """
    Abstract base class for an agent evaluator.

    """

    @abstractmethod
    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredAgentMessage]:
        """
        Executes the evaluator on the agent's context within the given budget.

        Args:
            context (AgentContext): The context for executing the evaluator.
            budget (Budget): The budget for evaluator execution.

        Returns:
            List[ScoredAgentMessage]: A list of scored agent messages resulting from the evaluation.

        Raises:
            None
        """
        pass
