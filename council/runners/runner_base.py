import abc
from collections.abc import Set
from concurrent import futures
import logging
from typing import List, Optional, Iterable

from council.contexts import ChainContext, ChatMessage
from . import RunnerResult, RunnerContext
from .budget import Budget
from .errrors import RunnerTimeoutError, RunnerError
from .runner_executor import RunnerExecutor

logger = logging.getLogger(__name__)


class RunnerBase(abc.ABC):
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

    @staticmethod
    def should_stop(context: ChainContext, budget: Budget, result: Optional[RunnerResult]) -> bool:
        if result and result.is_error:
            logger.debug('message="stopping" reason="skill error"')
            return True
        if budget.is_expired():
            logger.debug('message="stopping" reason="budget expired"')
        if context.cancellation_token.cancelled:
            logger.debug('message="stopping" reason="cancellation token is set"')
        return budget.is_expired() or context.cancellation_token.cancelled

    @staticmethod
    def make_new_context(context: ChainContext, messages: Iterable[ChatMessage]) -> "ChainContext":
        current = context.current.copy()
        current.extend(messages)
        histories = [*context.chain_histories[:-1], current]
        return ChainContext(context.chat_history, histories)

    @abc.abstractmethod
    def _run(
        self,
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        pass
