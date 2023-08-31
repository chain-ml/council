import abc
from collections.abc import Set
from concurrent import futures
import logging

from council.contexts import ChainContext
from .errrors import RunnerTimeoutError, RunnerError
from .runner_executor import RunnerExecutor
from ..monitors import Monitorable

logger = logging.getLogger(__name__)


class RunnerBase(Monitorable, abc.ABC):
    def run_from_chain_context(self, context: ChainContext, executor: RunnerExecutor):
        self.run(context, executor)

    """
    Base runner class that handles common execution logic, including error management and timeout
    """

    def run(
        self,
        context: ChainContext,
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
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        pass
