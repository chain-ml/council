import abc
from collections.abc import Set
from concurrent import futures
import logging

from council.contexts import ChainContext
from .runner_context import RunnerContext
from .budget import Budget
from .errrors import RunnerTimeoutError, RunnerError
from .runner_executor import RunnerExecutor

logger = logging.getLogger(__name__)


class RunnerBase(abc.ABC):
    def run_from_chain_context(self, chain_context: ChainContext, budget: Budget, executor: RunnerExecutor):
        context = RunnerContext(chain_context, budget)
        try:
            self.run(context, executor)
        finally:
            chain_context.current.extend(context.messages)

    """
    Base runner class that handles common execution logic, including error management and timeout
    """

    def run(
        self,
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        if context.should_stop():
            return

        logger.debug("start running %s", self.__class__.__name__)
        try:
            self._run(context, executor)
        except futures.TimeoutError as e:
            logger.debug("timeout running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerTimeoutError(self.__class__.__name__) from e
        except RunnerError:
            logger.debug("runner error running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise
        except Exception as e:
            logger.exception("an unexpected error occurred running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerError(f"an unexpected error occurred in {self.__class__.__name__}") from e
        finally:
            logger.debug("done running %s", self.__class__.__name__)

    @staticmethod
    def rethrow_if_exception(fs: Set[futures.Future]):
        [f.result(timeout=0) for f in fs]

    @abc.abstractmethod
    def _run(
        self,
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        pass
