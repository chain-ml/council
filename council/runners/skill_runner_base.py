import abc
import concurrent.futures

from council.contexts import ChainContext, SkillContext, IterationContext, ChatMessage
from . import RunnerTimeoutError, RunnerContext
from .runner_result import RunnerResult

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
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        self.run_skill(context, executor)

    def run_skill(self, context: RunnerContext, executor: RunnerExecutor) -> None:
        """
        Run the skill in a different thread, and await for completion
        """
        skill_context = SkillContext(context.make_chain_context(), IterationContext.empty())
        future = executor.submit(self.execute_skill, skill_context, context.budget)
        try:
            result = future.result(timeout=context.budget.duration)
            context.append(result)
        finally:
            future.cancel()

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext, budget: Budget) -> ChatMessage:
        """
        Run the skill in the current thread
        """
        pass
