from concurrent import futures
from council.contexts import ChainContext

from .budget import Budget
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Parallel(RunnerBase):
    """
    Runner that execution multiple :class:`.RunnerBase` in parallel
    """
    def __init__(self, *runners: RunnerBase):
        self.runners = runners

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        fs = [executor.submit(lambda: runner.run(context, budget, executor)) for runner in self.runners]
        try:
            dones, not_dones = futures.wait(fs, budget.remaining().duration, futures.FIRST_EXCEPTION)
            self.rethrow_if_exception(dones)
        finally:
            [f.cancel() for f in fs]
