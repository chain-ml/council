import abc

from council.contexts import ChainContext, ChatMessage, IterationContext, SkillContext

from ..utils import Option
from . import RunnerSkillError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class SkillRunnerBase(RunnerBase):
    """
    Runner that executes a :class:`.SkillBase`
    """

    def __init__(self, name: str) -> None:
        super().__init__("skill")
        self.monitor.name = name
        self._name = name

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        self.run_skill(context, executor)

    def run_skill(self, context: ChainContext, executor: RunnerExecutor) -> None:
        """
        Run the skill in a different thread, and await for completion
        """
        future = executor.submit(self.run_in_current_thread, context, IterationContext.empty())
        try:
            future.result(timeout=context.budget.remaining_duration)
        finally:
            future.cancel()

    def run_in_current_thread(self, context: ChainContext, iteration_context: Option[IterationContext]) -> None:
        """
        Run the skill in the current thread
        """
        try:
            with SkillContext.from_chain_context(context, iteration_context) as skill_context:
                message = self.execute_skill(skill_context)
                context.append(message)
        except Exception as e:
            context.logger.exception("unexpected error during execution of skill %s", self._name)
            context.append(self.from_exception(e))
            raise RunnerSkillError(f"an unexpected error occurred in skill {self._name}") from e

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext) -> ChatMessage:
        """
        Skill execution
        """
        pass

    def from_exception(self, exception: Exception) -> ChatMessage:
        message = f"skill '{self._name}' raised exception: {exception}"
        return ChatMessage.skill(message, data=None, source=self._name, is_error=True)
