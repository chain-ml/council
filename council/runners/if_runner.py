from council.contexts import ChatMessage, ChainContext

from .errrors import RunnerPredicateError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerPredicate


class If(RunnerBase):
    """
    Runner that executes only if the predicate returns `True`
    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        super().__init__()
        self.predicate = predicate
        self.runner = runner

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        try:
            result = self.predicate(context, context.budget.remaining())
        except Exception as e:
            context.append(ChatMessage.skill("IfRunner", f"predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e

        if result:
            self.runner.run(context, executor)
