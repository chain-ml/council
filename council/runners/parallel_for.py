import logging
from concurrent import futures
from typing import Iterable

from more_itertools import batched

from council.contexts import IterationContext, ChainContext
from council.utils import Option
from . import RunnerContext, Budget

from .errrors import RunnerGeneratorError
from .loop_runner_base import LoopRunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerGenerator
from .skill_runner_base import SkillRunnerBase

logger = logging.getLogger(__name__)


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

    def __init__(self, generator: RunnerGenerator, skill: SkillRunnerBase, parallelism: int = 5):
        """
        Initialize a new instance

        Parameters:
            generator(RunnerGenerator): a generator function that yields results

        """
        self._generator = generator
        self._skill = skill
        self._parallelism = parallelism

    def _run(self, context: RunnerContext, executor: RunnerExecutor) -> None:
        chain_context = context.make_chain_context()
        inner_contexts = []
        all_fs = []
        try:
            for batch in batched(self._generate(chain_context, context.budget.remaining()), self._parallelism):
                inner = [context.fork() for _ in batch]
                inner_contexts.extend(inner)
                fs = [executor.submit(self._run_skill, inner, iteration) for (inner, iteration) in zip(inner, batch)]
                all_fs.extend(fs)
                dones, not_dones = futures.wait(fs, context.budget.remaining().duration, futures.FIRST_EXCEPTION)
                self.rethrow_if_exception(dones)
        finally:
            [f.cancel() for f in all_fs]
            context.merge(inner_contexts)

    def _run_skill(self, context: RunnerContext, iteration: IterationContext):
        index = iteration.index
        logger.debug(f'message="start iteration" index="{index}"')
        try:
            self._skill.run_in_current_thread(context, Option.some(iteration))
        finally:
            logger.debug(f'message="end iteration" index="{index}"')

    def _generate(self, context: ChainContext, budget: Budget) -> Iterable[IterationContext]:
        try:
            for index, item in enumerate(self._generator(context, budget.remaining())):
                yield IterationContext(index, item)
        except Exception as e:
            raise RunnerGeneratorError from e
