import abc
from collections.abc import Set
from concurrent import futures

from council.contexts import ChainContext, Monitorable, Monitored
from .errrors import RunnerError, RunnerTimeoutError
from .runner_executor import RunnerExecutor


class RunnerBase(Monitorable, abc.ABC):
    def run_from_chain_context(self, context: ChainContext, executor: RunnerExecutor):
        self.run(context, executor)

    """
    Base runner class that handles common execution logic, including error management and timeout
    """

    def fork_run_merge(self, runner: Monitored["RunnerBase"], context: ChainContext, executor: RunnerExecutor):
        inner = context.fork_for(runner)
        try:
            runner.inner.run(inner, executor)
        finally:
            context.merge([inner])

    def run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        if context.should_stop():
            return

        context.logger.debug("start running %s", self.__class__.__name__)
        try:
            with context:
                self._run(context, executor)
        except futures.TimeoutError as e:
            context.logger.debug("timeout running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerTimeoutError(self.__class__.__name__) from e
        except RunnerError:
            context.logger.debug("runner error running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise
        except Exception as e:
            context.logger.exception("an unexpected error occurred running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerError(f"an unexpected error occurred in {self.__class__.__name__}") from e
        finally:
            context.logger.debug("done running %s", self.__class__.__name__)

    @staticmethod
    def rethrow_if_exception(fs: Set[futures.Future]):
        [f.result(timeout=0) for f in fs]

    @abc.abstractmethod
    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        pass
