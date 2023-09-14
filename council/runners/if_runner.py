from typing import Optional

from council.contexts import ChatMessage, ChainContext

from .errrors import RunnerPredicateError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerPredicate


class If(RunnerBase):
    """
    Runner that executes only if the predicate returns `True`
    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase, else_runner: Optional[RunnerBase] = None):
        """
        Args:
            predicate: a predicate function
            runner: a runner to be executed only if the predicate returns `True`
            else_runner: an optional runner to be executed only if the predicate returns `False`
        """
        super().__init__("ifRunner")
        self._predicate = predicate
        self._then = self.new_monitor("then", runner)
        self._maybe_else = self.new_monitor("else", else_runner) if else_runner is not None else None

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        try:
            result = self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("IfRunner", f"predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e

        if result:
            self._then.inner.run(context, executor)
        elif self._maybe_else is not None:
            self._maybe_else.inner.run(context, executor)
