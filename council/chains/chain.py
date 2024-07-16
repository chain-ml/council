from typing import Optional, Sequence

from council.chains.chain_base import ChainBase
from council.contexts import ChainContext, Monitored
from council.runners import RunnerBase, RunnerExecutor, Sequential


class Chain(ChainBase):
    """
    Represents a chain of skills that can be executed in a specific order.
    """

    def __init__(self, name: str, description: str, runners: Sequence[RunnerBase], support_instructions: bool = False):
        """
        Initializes the Chain object.

        Args:
            name (str): The name of the chain.
            description (str): The description of the chain.
            runners (List[RunnerBase]): The list of runners representing the skills in the chain.

        Raises:
            None
        """
        super().__init__(name, description, support_instructions)
        self._runner: Monitored[RunnerBase] = self.new_monitor("runner", Sequential.from_list(runners))

    @property
    def runner(self) -> RunnerBase:
        """
        the runner of the chain
        """
        return self._runner.inner

    def _execute(
        self,
        context: ChainContext,
        executor: Optional[RunnerExecutor] = None,
    ) -> None:
        executor = (
            RunnerExecutor(max_workers=10, thread_name_prefix=f"chain_{self.name}") if executor is None else executor
        )

        self._runner.inner.fork_run_merge(self._runner, context, executor)
