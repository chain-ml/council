import abc
import concurrent.futures
import logging

from council.contexts import ChainContext, SkillContext, IterationContext, ChatMessage
from . import RunnerTimeoutError, RunnerContext, RunnerSkillError
from .runner_result import RunnerResult

from .budget import Budget
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


logger = logging.getLogger(__name__)


class SkillRunnerBase(RunnerBase):
    """
    Runner that executes a :class:`.SkillBase`
    """

    def __init__(self, name):
        self._name = name

    def _run(
        self,
        context: RunnerContext,
        executor: RunnerExecutor,
    ) -> None:
        self.run_skill(context, executor)

    def run_skill(self, context: RunnerContext, executor: RunnerExecutor) -> None:
        """
        Run the skill in a different thread, and await for completion
        """
        skill_context = SkillContext(context.make_chain_context(), IterationContext.empty())
        future = executor.submit(self.execute_skill, skill_context, context.budget)
        try:
            result = future.result(timeout=context.budget.remaining().duration)
            context.append(result)
        except concurrent.futures.TimeoutError:
            raise
        except Exception as e:
            logger.exception("unexpected error during execution of skill %s", self._name)
            message = self.from_exception(e)
            context.append(message)
            raise RunnerSkillError(f"an unexpected error occurred in skill {self._name}") from e

        finally:
            future.cancel()

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext, budget: Budget) -> ChatMessage:
        """
        Run the skill in the current thread
        """
        pass

    def from_exception(self, exception: Exception) -> ChatMessage:
        message = f"skill '{self._name}' raised exception: {exception}"
        return ChatMessage.skill(message, data=None, source=self._name, is_error=True)

