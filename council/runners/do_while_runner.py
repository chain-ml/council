from council.contexts import ChainContext, ChatMessage
from council.runners import RunnerBase, RunnerPredicate, RunnerExecutor, RunnerPredicateError


class DoWhile(RunnerBase):
    """
    Runner that executes an inner Runner while the given predicate returns `True`.
    The predicate is executed at the end of the loop. As such, the inner runner executes at least once.
    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        """
        Args:
            predicate: a predicate function
            runner: a runner to be executed while the predicate returns `True`
        """
        super().__init__("doWhileRunner")
        self._predicate = predicate
        self._body = self.new_monitor("doWhileBody", runner)

    def _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        while True:
            self._body.inner.run(context, executor)

            if not self.check_predicate(context):
                return

    def check_predicate(self, context: ChainContext) -> bool:
        try:
            return self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("DoWhileRunner", f"Predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e
