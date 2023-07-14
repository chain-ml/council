import abc
from council.contexts import ChainContext, SkillContext, IterationContext

from .budget import Budget
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class SkillRunnerBase(RunnerBase):
    """
    Runner that executes a :class:`.SkillBase`
    """
    def __init__(self, name):
        self._name = name

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        self.run_skill(SkillContext(context, IterationContext.empty()), budget, executor)

    def run_skill(self, context: SkillContext, budget: Budget, executor: RunnerExecutor) -> None:
        """
        Run the skill in a different thread, and await for completion
        """
        future = executor.submit(self.execute_skill, context, budget)
        try:
            future.result(timeout=budget.duration)
        finally:
            future.cancel()

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext, budget: Budget) -> None:
        """
        Run the skill in the current thread
        """
        pass
