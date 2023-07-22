import abc
import logging

from council.contexts import SkillContext, IterationContext, ChatMessage
from . import RunnerContext, RunnerSkillError

from .budget import Budget
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from ..utils import Option

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
        future = executor.submit(self.run_in_current_thread, context, IterationContext.empty())
        try:
            future.result(timeout=context.budget.remaining().duration)
        finally:
            future.cancel()

    def run_in_current_thread(self, context: RunnerContext, iteration_context: Option[IterationContext]) -> None:
        """
        Run the skill in the current thread
        """
        try:
            skill_context = SkillContext(context.make_chain_context(), iteration_context)
            message = self.execute_skill(skill_context, context.budget.remaining())
            context.append(message)
        except Exception as e:
            logger.exception("unexpected error during execution of skill %s", self._name)
            context.append(self.from_exception(e))
            raise RunnerSkillError(f"an unexpected error occurred in skill {self._name}") from e

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext, budget: Budget) -> ChatMessage:
        """
        Skill execution
        """
        pass

    def from_exception(self, exception: Exception) -> ChatMessage:
        message = f"skill '{self._name}' raised exception: {exception}"
        return ChatMessage.skill(message, data=None, source=self._name, is_error=True)
