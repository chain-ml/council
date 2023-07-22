import logging
from concurrent import futures
from typing import Iterable

from more_itertools import batched

from council.contexts import ChainContext, SkillContext, IterationContext
from council.utils import Option

from .budget import Budget
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

    def _run(self, context: ChainContext, budget: Budget, executor: RunnerExecutor) -> None:
        for batch in batched(self._generate(context, budget), self._parallelism):
            fs = [executor.submit(self._run_skill, context, item, budget) for item in batch]
            try:
                dones, not_dones = futures.wait(fs, budget.remaining().duration, futures.FIRST_EXCEPTION)
                self.rethrow_if_exception(dones)
            finally:
                [f.cancel() for f in fs]

    def _run_skill(self, context: ChainContext, iteration: IterationContext, budget: Budget):
        index = iteration.index
        logger.debug(f'message="start iteration" index="{index}"')
        self._skill.execute_skill(SkillContext(context, Option.some(iteration)), budget)
        logger.debug(f'message="end iteration" index="{index}"')

    def _generate(self, chain_context: ChainContext, budget: Budget) -> Iterable[IterationContext]:
        try:
            for index, item in enumerate(self._generator(chain_context, budget)):
                yield IterationContext(index, item)
        except Exception as e:
            raise RunnerGeneratorError from e
