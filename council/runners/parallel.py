from concurrent import futures

from .runner_context import RunnerContext
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
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        fs = [executor.submit(lambda: self._run_one(runner, context, executor)) for runner in self.runners]
        try:
            dones, not_dones = futures.wait(fs, context.budget.remaining().duration, futures.FIRST_EXCEPTION)
            self.rethrow_if_exception(dones)
        finally:
            [f.cancel() for f in fs]

    def _run_one(self, runner: RunnerBase, context: RunnerContext, executor: RunnerExecutor):
        inner = context.fork()
        try:
            runner.run(inner, executor)
        finally:
            context.merge([inner])
