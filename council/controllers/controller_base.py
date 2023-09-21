from abc import ABC, abstractmethod
from typing import List, Sequence

from council.chains import ChainBase
from council.contexts import AgentContext, Monitorable
from .execution_unit import ExecutionUnit


class ControllerBase(Monitorable, ABC):
    """
    Abstract base class for an agent controller.
    """

    def __init__(self, chains: Sequence[ChainBase]):
        """
        Args:
            chains (List[Chain]): The list of chains available for execution.
        """
        super().__init__("controller")
        self._chains = list(chains)

    def execute(self, context: AgentContext) -> List[ExecutionUnit]:
        """
        Generates an execution plan for the agent based on the provided context, chains, and budget.

        Args:
            context (AgentContext): The context for generating the execution plan.

        Returns:
            List[ExecutionUnit]: A list of execution units representing the execution plan.

        Raises:
            None
        """
        with context:
            return self._execute(context)

    @abstractmethod
    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        pass

    @property
    def chains(self) -> Sequence[ChainBase]:
        """
        the chains of the controller
        """
        return self._chains
