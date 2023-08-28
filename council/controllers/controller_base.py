from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from council.chains import Chain
from council.contexts import AgentContext, ScoredChatMessage
from council.runners import Budget
from .execution_unit import ExecutionUnit
from council.monitors import Monitorable


class ControllerBase(Monitorable, ABC):
    """
    Abstract base class for an agent controller.

    """

    def __init__(self):
        super().__init__()

    def monitor_get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        with context.new_log_entry():
            return self.get_plan(context, chains, budget)

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

    def monitor_select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        with context.new_log_entry():
            return self.select_responses(context)

    @abstractmethod
    def select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        """
        Selects responses from the agent's context.

        Args:
            context (AgentContext): The context for selecting responses.

        Returns:
            List[ScoredChatMessage]: A list of scored agent messages representing the selected responses.

        Raises:
            None
        """
        pass
