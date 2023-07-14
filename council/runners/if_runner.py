from council.core import ChainContext
from council.core.execution_context import SkillErrorMessage

from .budget import Budget
from .errrors import RunnerPredicateError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerPredicate


class If(RunnerBase):
    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        self.predicate = predicate
        self.runner = runner

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        try:
            result = self.predicate(context, budget)
        except Exception as e:
            context.current.messages.append(SkillErrorMessage("IfRunner", f"predicate raised exception: {e}"))
            raise RunnerPredicateError from e
        if result:
            self.runner.run(context, budget, executor)
