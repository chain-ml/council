from typing import List, Sequence

from council.contexts import AgentContext

from ..chains import ChainBase
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit


class BasicController(ControllerBase):
    """a basic controller that requests all chains to be executed and returns all results"""

    def __init__(self, chains: Sequence[ChainBase], parallelism: bool = False) -> None:
        super().__init__(chains, parallelism)

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        return [ExecutionUnit(chain, context.budget, rank=self.default_execution_unit_rank) for chain in self._chains]
