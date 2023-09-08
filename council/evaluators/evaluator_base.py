from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, Monitorable, ScoredChatMessage


class EvaluatorBase(Monitorable, ABC):
    """
    Abstract base class for an agent evaluator.

    """

    def __init__(self):
        super().__init__("evaluator")

    def execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Executes the evaluator on the agent's context within the given budget.

        Args:
            context (AgentContext): The context for executing the evaluator.

        Returns:
            List[ScoredChatMessage]: A list of scored agent messages resulting from the evaluation.

        Raises:
            None
        """
        with context.log_entry:
            return self._execute(context=context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        pass
