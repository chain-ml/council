from abc import ABC, abstractmethod
from typing import List

from council.contexts import AgentContext, ScoredChatMessage
from council.monitors import Monitorable
from council.runners import Budget


class EvaluatorBase(Monitorable, ABC):
    """
    Abstract base class for an agent evaluator.

    """

    def __init__(self):
        super().__init__()

    def monitor_execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        with context.new_log_entry():
            return self.execute(context, budget)

    @abstractmethod
    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        """
        Executes the evaluator on the agent's context within the given budget.

        Args:
            context (AgentContext): The context for executing the evaluator.
            budget (Budget): The budget for evaluator execution.

        Returns:
            List[ScoredChatMessage]: A list of scored agent messages resulting from the evaluation.

        Raises:
            None
        """
        pass
