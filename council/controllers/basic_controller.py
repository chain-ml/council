from typing import List

from council.contexts import AgentContext, Budget
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit


class BasicController(ControllerBase):
    """a basic controller that requests all chains to be executed and returns all results"""

    def _execute(self, context: AgentContext, budget: Budget) -> List[ExecutionUnit]:
        return [ExecutionUnit(chain, budget) for chain in self._chains]
