from concurrent import futures
from typing import Iterable

from council.contexts import ChainContext, IterationContext
from council.utils import Option
from more_itertools import batched

from .errors import RunnerGeneratorError
from .loop_runner_base import LoopRunnerBase
from .runner_executor import RunnerExecutor
from .skill_runner_base import SkillRunnerBase
from .types import RunnerGenerator


class ParallelFor(LoopRunnerBase):
    """
    Invoke a given skill for each value returned by a given generator function.
    Can run multiple iteration in parallel.
    For each invocation, the current iteration current is provided through the skill context
    :meth:`.SkillContext.iteration`.

    :meth:`.IterationContext.value` provides the value as returned by the generator function

    :meth:`.IterationContext.index` provides the index of the iteration

    Notes:
        Skill iteration are scheduled in the order given by the generator function.
        However, because multiple iterations can execute in parallel, no assumptions should be made on
        the order of results.
    """

    def __init__(self, generator: RunnerGenerator, skill: SkillRunnerBase, parallelism: int = 5) -> None:
        """
        Initialize a new instance

        Parameters:
            generator(RunnerGenerator): a generator function that yields results

        """
        super().__init__("parallelForRunner")
        self._generator = generator
        self._skill = self.new_monitor("skill", skill)
        self._parallelism = parallelism

    def _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        inner_contexts = []
        all_fs = []
        try:
            for batch in batched(self._generate(context), self._parallelism):
                inner = [context.fork_for(self._skill) for _ in batch]
                inner_contexts.extend(inner)
                fs = [executor.submit(self._run_skill, inner, iteration) for (inner, iteration) in zip(inner, batch)]
                all_fs.extend(fs)
                dones, not_dones = futures.wait(fs, context.budget.remaining_duration, futures.FIRST_EXCEPTION)
                self.rethrow_if_exception(dones)
        finally:
            [f.cancel() for f in all_fs]
            context.merge(inner_contexts)

    def _run_skill(self, context: ChainContext, iteration: IterationContext) -> None:
        index = iteration.index
        context.logger.debug(f'message="start iteration" index="{index}"')
        try:
            self._skill.inner.run_in_current_thread(context, Option.some(iteration))
        finally:
            context.logger.debug(f'message="end iteration" index="{index}"')

    def _generate(self, context: ChainContext) -> Iterable[IterationContext]:
        try:
            for index, item in enumerate(self._generator(context)):
                yield IterationContext(index, item)
        except Exception as e:
            raise RunnerGeneratorError from e
