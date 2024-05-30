from typing import Sequence

from council.contexts import ChainContext

from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Sequential(RunnerBase):
    """
    Runner that executes a list of :class:`.RunnerBase` in sequence
    """

    def __init__(self, *runners: RunnerBase) -> None:
        super().__init__("sequenceRunner")
        self._runners = self.new_monitors("sequence", runners)

    def _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        for runner in self._runners:
            if context.should_stop():
                return

            self.fork_run_merge(runner, context, executor)

    @staticmethod
    def from_list(runners: Sequence[RunnerBase]) -> RunnerBase:
        if len(runners) == 1:
            return runners[0]

        return Sequential(*runners)
