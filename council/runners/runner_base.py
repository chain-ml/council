import abc
from collections.abc import Set
from concurrent import futures
import logging

from council.core import ChainContext
from .budget import Budget
from .errrors import RunnerTimeoutError, RunnerError
from .runner_executor import RunnerExecutor

logger = logging.getLogger(__name__)


class RunnerBase(abc.ABC):
    def run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        if self.should_stop(context, budget):
            return

        logger.debug("start running %s", self.__class__.__name__)
        try:
            return self._run(context, budget, executor)
        except futures.TimeoutError as e:
            logger.debug("timeout running %s", self.__class__.__name__)
            context.cancellationToken.cancel()
            raise RunnerTimeoutError(self.__class__.__name__) from e
        except RunnerError:
            context.cancellationToken.cancel()
            raise
        except Exception as e:
            logger.exception("an unexpected error occurred running %s", self.__class__.__name__)
            context.cancellationToken.cancel()
            raise RunnerError(f"an unexpected error occurred in {self.__class__.__name__}") from e
        finally:
            logger.debug("done running %s", self.__class__.__name__)

    @staticmethod
    def rethrow_if_exception(fs: Set[futures.Future]):
        [f.result(timeout=0) for f in fs]

    @staticmethod
    def should_stop(context: ChainContext, budget: Budget) -> bool:
        if budget.is_expired():
            logger.debug('message="stopping" reason="budget expired"')
        if context.cancellationToken.cancelled:
            logger.debug('message="stopping" reason="cancellation token is set"')
        return budget.is_expired() or context.cancellationToken.cancelled

    @abc.abstractmethod
    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        pass
