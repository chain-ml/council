from council.contexts import ChainContext

from .budget import Budget
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Sequential(RunnerBase):
    """
    Runner that executes a list of :class:`.RunnerBase` in sequence
    """

    def __init__(self, *runners: RunnerBase):
        self.runners = runners

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        for runner in self.runners:
            if self.should_stop(context, budget):
                return
            runner.run(context, budget.remaining(), executor)

    @staticmethod
    def from_list(*runners: RunnerBase) -> RunnerBase:
        if len(runners) == 1:
            return runners[0]

        return Sequential(*runners)
