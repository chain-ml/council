from typing import List

from council.contexts import AgentContext
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit


class BasicController(ControllerBase):
    """a basic controller that requests all chains to be executed and returns all results"""

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        return [ExecutionUnit(chain, context.budget) for chain in self._chains]
