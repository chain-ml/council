import abc
import logging
from concurrent import futures
from typing import Callable, Any, Set, Iterable

from more_itertools import batched

from council.core.budget import Budget
from council.core.execution_context import (
    ChainContext,
    SkillErrorMessage,
    SkillContext,
    IterationContext,
)
from council.utils import Option

RunnerExecutor = futures.ThreadPoolExecutor
logger = logging.getLogger(__name__)

RunnerPredicate = Callable[[ChainContext, Budget], bool]
RunnerGenerator = Callable[[ChainContext, Budget], Iterable[Any]]


class RunnerError(Exception):
    pass


class RunnerTimeoutError(RunnerError):
    pass


class RunnerSkillError(RunnerError):
    pass


class RunnerPredicateError(RunnerError):
    pass


class RunnerGeneratorError(RunnerError):
    pass


def new_runner_executor(name: str = "skill_runner") -> RunnerExecutor:
    return RunnerExecutor(thread_name_prefix=name, max_workers=10)


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


class SkillRunnerBase(RunnerBase):
    def __init__(self, name):
        self._name = name

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        self.run_skill(SkillContext(context, IterationContext.empty()), budget, executor)

    def run_skill(self, context: SkillContext, budget: Budget, executor: RunnerExecutor) -> None:
        """
        Run the skill in a different thread, and await for completion
        """
        future = executor.submit(self.execute_skill, context, budget)
        try:
            future.result(timeout=budget.duration)
        finally:
            future.cancel()

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext, budget: Budget) -> None:
        """
        Run the skill in the current thread
        """
        pass


class Sequential(RunnerBase):
    def __init__(self, *runners: RunnerBase):
        self.runners = runners

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        for runner in self.runners:
            if self.should_stop(context, budget):
                return
            runner.run(context, budget.remaining(), executor)

    @staticmethod
    def from_list(*runners: RunnerBase) -> RunnerBase:
        if len(runners) == 1:
            return runners[0]

        return Sequential(*runners)


class Parallel(RunnerBase):
    def __init__(self, *runners: RunnerBase):
        self.runners = runners

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        fs = [executor.submit(lambda: runner.run(context, budget, executor)) for runner in self.runners]
        try:
            dones, not_dones = futures.wait(fs, budget.remaining().duration, futures.FIRST_EXCEPTION)
            self.rethrow_if_exception(dones)
        finally:
            [f.cancel() for f in fs]


class If(RunnerBase):
    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        self.predicate = predicate
        self.runner = runner

    def _run(
        self,
        context: ChainContext,
        budget: Budget,
        executor: RunnerExecutor,
    ) -> None:
        try:
            result = self.predicate(context, budget)
        except Exception as e:
            context.current.messages.append(SkillErrorMessage("IfRunner", f"predicate raised exception: {e}"))
            raise RunnerPredicateError from e
        if result:
            self.runner.run(context, budget, executor)


class LoopRunnerBase(RunnerBase, abc.ABC):
    pass


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
