from council.contexts import ChainContext, ChatMessage
from . import RunnerResult

from .budget import Budget
from .errrors import RunnerPredicateError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerPredicate


class If(RunnerBase):
    """
    Runner that executes only if the predicate returns `True`
    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        self.predicate = predicate
        self.runner = runner

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> RunnerResult:
        try:
            result = self.predicate(context, budget)
        except Exception as e:
            context.current.append(ChatMessage.skill("IfRunner", f"predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e
        if result:
            return self.runner.run(context, budget, executor)
        return RunnerResult()
