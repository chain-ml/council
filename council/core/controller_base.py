from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from council.core.budget import Budget
from council.core.chain import Chain
from council.core.execution_context import AgentContext, ScoredAgentMessage
from council.core.execution_unit import ExecutionUnit


class ControllerBase(ABC):
    """
    Abstract base class for an agent controller.

    """

    @abstractmethod
    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        """
        Generates an execution plan for the agent based on the provided context, chains, and budget.

        Args:
            context (AgentContext): The context for generating the execution plan.
            chains (List[Chain]): The list of chains available for execution.
            budget (Budget): The budget for agent execution.

        Returns:
            List[ExecutionUnit]: A list of execution units representing the execution plan.

        Raises:
            None
        """
        pass

    @abstractmethod
    def select_responses(self, context: AgentContext) -> List[ScoredAgentMessage]:
        """
        Selects responses from the agent's context.

        Args:
            context (AgentContext): The context for selecting responses.

        Returns:
            List[ScoredAgentMessage]: A list of scored agent messages representing the selected responses.

        Raises:
            None
        """
        pass
