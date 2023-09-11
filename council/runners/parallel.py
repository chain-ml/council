from concurrent import futures
from council.contexts import ChainContext

from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Parallel(RunnerBase):
    """
    Runner that execution multiple :class:`.RunnerBase` in parallel
    """

    def __init__(self, *runners: RunnerBase):
        super().__init__("parallelRunner")
        self._runners = self.new_monitors("parallel", runners)

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        contexts = [(runner.inner, context.fork_for(runner)) for runner in self._runners]

        # Seems like it is a bad idea using lambda as the function in submit,
        # which results into inconsistent invocation (wrong arguments)
        fs = [executor.submit(runner.run, inner, executor) for (runner, inner) in contexts]
        try:
            dones, not_dones = futures.wait(fs, context.budget.remaining_duration, futures.FIRST_EXCEPTION)
            self.rethrow_if_exception(dones)
        finally:
            context.merge([context for (_, context) in contexts])
            [f.cancel() for f in fs]
