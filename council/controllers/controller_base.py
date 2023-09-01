from abc import ABC, abstractmethod
from typing import List, Sequence

from council.chains import Chain
from council.contexts import AgentContext, Budget
from .execution_unit import ExecutionUnit
from council.monitors import Monitorable


class ControllerBase(Monitorable, ABC):
    """
    Abstract base class for an agent controller.

    """

    def __init__(self, chains: List[Chain]):
        """
        Args:
            chains (List[Chain]): The list of chains available for execution.
        """
        super().__init__()
        self._chains = chains

    def execute(self, context: AgentContext, budget: Budget) -> List[ExecutionUnit]:
        """
        Generates an execution plan for the agent based on the provided context, chains, and budget.

        Args:
            context (AgentContext): The context for generating the execution plan.
            budget (Budget): The budget for agent execution.

        Returns:
            List[ExecutionUnit]: A list of execution units representing the execution plan.

        Raises:
            None
        """
        with context:
            return self._execute(context, budget)

    @abstractmethod
    def _execute(self, context: AgentContext, budget: Budget) -> List[ExecutionUnit]:
        pass

    @property
    def chains(self) -> Sequence[Chain]:
        return self._chains
