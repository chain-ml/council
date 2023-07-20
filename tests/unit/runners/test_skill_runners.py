import time
import unittest
from typing import List, Any

from council.contexts import (
    ChatHistory,
    ChainContext,
    ChainHistory,
    SkillContext,
    ChatMessage,
)

from council.skills.skill_base import SkillBase
from council.runners import (
    Sequential,
    RunnerBase,
    Parallel,
    new_runner_executor,
    RunnerTimeoutError,
    If,
    ParallelFor,
    RunnerPredicateError,
    RunnerGeneratorError,
    RunnerSkillError,
    Budget, RunnerContext,
)


class MySkillException(Exception):
    pass


class SkillTest(SkillBase):
    def __init__(self, name: str, wait: float):
        super().__init__(name)
        self.wait = wait

    def execute(self, context: SkillContext, budget: Budget) -> ChatMessage:
        time.sleep(abs(self.wait))
        if self.wait < 0:
            raise MySkillException("invalid wait")
        if context.iteration.map_or(lambda i: i.value, 0.0) < 0:
            raise MySkillException("invalid iteration")
        return self.build_success_message(self._name, context.iteration.map_or(lambda i: i.value, -1))


class TestSkillRunners(unittest.TestCase):
    def setUp(self) -> None:
        self.history = ChatHistory()
        self.context = ChainContext(ChatHistory(), [ChainHistory()])
        self.executor = new_runner_executor(name="test_skill_runner")

    def _execute(self, runner: RunnerBase, budget: Budget) -> None:
        runner.run_from_chain_context(self.context, budget, self.executor)

    def assertSuccessMessages(self, expected: List[str]):
        self.assertEqual(
            expected,
            [m.message for m in self.context.current.messages if m.is_kind_skill and m.is_ok],
        )

    def test_one_skill(self):
        instance = Sequential(SkillTest("single", 0.1))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["single"])

    def test_one_skill_timeout(self):
        instance = Sequential(SkillTest("single", 0.2))
        with self.assertRaises(RunnerTimeoutError):
            self._execute(instance, Budget(0.1))
            self.assertSuccessMessages([])
            time.sleep(0.2)
            self.assertSuccessMessages([])

    def test_sequence(self):
        instance = Sequential(SkillTest("first", 0.2), SkillTest("second", 0.1))
        self._execute(instance, Budget(0.35))
        self.assertSuccessMessages(["first", "second"])

    def test_sequence_timeout(self):
        instance = Sequential(SkillTest("first", 0.2), SkillTest("second", 0.1))
        with self.assertRaises(RunnerTimeoutError):
            self._execute(instance, Budget(0.25))
        self.assertSuccessMessages(["first"])

        time.sleep(0.1)
        self.assertSuccessMessages(["first"])

    def test_sequence_with_exception(self):
        instance = Sequential(SkillTest("first", 0.3), SkillTest("second", -0.2), SkillTest("third", 0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(1000))

        time.sleep(1)
        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages(["first"])
        self.assertTrue(self.context.current.try_last_message.unwrap().is_error)

    def test_parallel(self):
        instance = Parallel(SkillTest("first", 0.2), SkillTest("second", 0.1))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["second", "first"])

    def test_parallel_sequence(self):
        sequence = Sequential(SkillTest("second", 0.1), SkillTest("third", 0.2))
        parallel = Parallel(sequence, SkillTest("first", 0.15), SkillTest("fourth", 0.4))
        self._execute(parallel, Budget(1))
        self.assertFalse(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["first", "second", "third", "fourth"])

    def test_parallel_with_exception(self):
        instance = Parallel(SkillTest("first", .3), SkillTest("second", -.2), SkillTest("third", .1))
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertTrue(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["third"])
        self.assertTrue(self.context.current.try_last_message.unwrap().is_error)

        time.sleep(.2)
        self.assertSuccessMessages(["third"])

    def test_if_runner_true(self):
        instance = Sequential(If(lambda a, b: True, SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["maybe", "always"])

    def test_if_runner_true_with_error(self):
        instance = Sequential(If(lambda a, b: True, SkillTest("maybe", -0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages([])

    def test_if_runner_predicate_throw(self):
        def predicate_throw(a, b):
            raise Exception("predicate")

        instance = Sequential(If(lambda a, b: predicate_throw(a, b), SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerPredicateError):
            self._execute(instance, Budget(1))
        self.assertSuccessMessages([])

    def test_if_runner_false(self):
        instance = Sequential(If(lambda a, b: False, SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["always"])

    def test_sequence_if_predicate_with_context_runner(self):
        def predicate(chain_context: ChainContext, budget: Budget):
            return chain_context.current.messages[-1].message == "first"

        instance = Sequential(SkillTest("first", 0.1), If(predicate, SkillTest("maybe", 0.2)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["first", "maybe", "always"])

    def test_parallel_for(self):
        count = 100

        def generator(chain_context: ChainContext, budget: Budget) -> Any:
            for i in range(count):
                yield i

        instance = ParallelFor(generator, SkillTest("for each", 0.01))
        self._execute(instance, Budget(2))
        self.assertSuccessMessages(["for each" for i in range(count)])
        data = [m.data for m in self.context.current.messages if m.is_ok]
        data.sort()
        self.assertEqual([i for i in range(count)], data)

    def test_parallel_for_last_throw(self):
        def generator(chain_context: ChainContext, budget: Budget):
            for _ in [1, 2, 3, 4]:
                yield 0.01
            yield -0.01

        instance = ParallelFor(generator, SkillTest("for each", 0.01), parallelism=1)
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(0.5))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages(["for each" for i in range(4)])

    def test_parallel_for_generator_throw(self):
        class MyGeneratorError(Exception):
            pass

        def generator(chain_context: ChainContext, budget: Budget):
            yield 1
            raise MyGeneratorError("my error")

        instance = ParallelFor(generator, SkillTest("for each", 0.01), parallelism=2)
        with self.assertRaises(RunnerGeneratorError) as cm:
            self._execute(instance, Budget(0.5))

        self.assertIsInstance(cm.exception.__cause__, MyGeneratorError)
