from .runner_context import RunnerContext
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
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        for runner in self.runners:
            if context.should_stop():
                return
            runner.run(context, executor)

    @staticmethod
    def from_list(*runners: RunnerBase) -> RunnerBase:
        if len(runners) == 1:
            return runners[0]

        return Sequential(*runners)
