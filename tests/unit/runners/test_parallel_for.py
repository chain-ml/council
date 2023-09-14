from typing import Any

from council.contexts import Budget, ChainContext, Consumption
from council.runners import (
    ParallelFor,
    RunnerGeneratorError,
    RunnerSkillError,
)
from .helpers import MySkillException, RunnerTestCase, SkillTest


class TestSkillRunners(RunnerTestCase):
    def test_parallel_for(self):
        count = 100

        def generator(chain_context: ChainContext) -> Any:
            for i in range(count):
                yield i

        instance = ParallelFor(generator, SkillTest("for each", 0.01))
        self.execute(instance, Budget(2))
        self.assertSuccessMessages(["for each" for i in range(count)])
        data = [m.data for m in self.context.current.messages if m.is_ok]
        self.assertEqual([i for i in range(count)], data)

    def test_parallel_for_last_throw(self):
        def generator(chain_context: ChainContext):
            for _ in [1, 2, 3, 4]:
                yield 0.01
            yield -0.01

        instance = ParallelFor(generator, SkillTest("for each", 0.01), parallelism=1)
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(0.5))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages(["for each" for i in range(4)])

    def test_parallel_for_generator_throw(self):
        class MyGeneratorError(Exception):
            pass

        def generator(chain_context: ChainContext):
            yield 1
            raise MyGeneratorError("my error")

        instance = ParallelFor(generator, SkillTest("for each", 0.01), parallelism=2)
        with self.assertRaises(RunnerGeneratorError) as cm:
            self.execute(instance, Budget(0.5))

        self.assertIsInstance(cm.exception.__cause__, MyGeneratorError)

    def test_parallel_for_generator_consume_budget(self):
        count = 10

        def generator(chain_context: ChainContext) -> Any:
            for i in range(count):
                chain_context.budget.add_consumption(1, "unit", "budget")
                yield i

        instance = ParallelFor(generator, SkillTest("for each", 0.01))
        self.execute(instance, Budget(2, limits=[Consumption(20, "unit", "budget")]))
        self.assertSuccessMessages(["for each" for i in range(count)])
        data = [m.data for m in self.context.current.messages if m.is_ok]
        self.assertEqual([i for i in range(count)], data)
        self.assertEqual(self.context.budget._remaining[0].value, 10)
