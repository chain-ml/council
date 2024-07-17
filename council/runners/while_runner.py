from council.contexts import ChainContext, ChatMessage
from council.runners import RunnerBase, RunnerExecutor, RunnerPredicate, RunnerPredicateError


class While(RunnerBase):
    """
    Runner that executes while the given predicate returns `True`
    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase) -> None:
        """
        Args:
            predicate: a predicate function evaluated in the while loop
            runner: a runner to be executed while the predicate returns `True`
        """
        super().__init__("whileRunner")
        self._predicate = predicate
        self._body = self.new_monitor("whileBody", runner)

    def _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        while self.check_predicate(context):
            self._body.inner.run(context, executor)

    def check_predicate(self, context: ChainContext) -> bool:
        try:
            return self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("WhileRunner", f"Predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e
