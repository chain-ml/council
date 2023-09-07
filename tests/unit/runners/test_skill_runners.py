import json
import time
import unittest
from typing import List, Any, Optional

from council import monitors
from council.contexts import (
    ChainContext,
    Consumption,
    SkillContext,
    ChatMessage,
    Budget,
    AgentContext,
)
from council.mocks import MockMonitored

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
)


class MySkillException(Exception):
    pass


class SkillTest(SkillBase):
    def __init__(self, name: str, wait: float, budget_kind: Optional[str] = None):
        super().__init__(name)
        self.wait = wait
        self.budget_kind = budget_kind

    def execute(self, context: SkillContext) -> ChatMessage:
        time.sleep(abs(self.wait))
        if self.wait < 0:
            raise MySkillException("invalid wait")
        if context.iteration.map_or(lambda i: i.value, 0.0) < 0:
            raise MySkillException("invalid iteration")
        if self.budget_kind is not None:
            context.budget.add_consumption(Consumption(1, "unit", self.budget_kind))
        return self.build_success_message(self._name, context.iteration.map_or(lambda i: i.value, -1))


class SkillTestAppend(SkillBase):
    def execute(self, context: SkillContext) -> ChatMessage:
        message = context.current.try_last_message.map_or(lambda m: m.message, "")
        return self.build_success_message(message + self.name)


class SkillTestMerge(SkillBase):
    def __init__(self, from_skills: list[str]):
        super().__init__("merge")
        self.from_skills = from_skills

    def execute(self, context: SkillContext) -> ChatMessage:
        message = "".join([context.try_last_message_from_skill(name).unwrap().message for name in self.from_skills])
        return self.build_success_message(message)


class TestSkillRunners(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = new_runner_executor(name="test_skill_runner")

    def _execute(self, runner: RunnerBase, budget: Budget) -> None:
        # print(f"\n{monitors.render_as_text(runner)}")
        print(f"\n{monitors.render_as_json(runner)}")
        context = AgentContext.empty()
        context.new_iteration()
        with context.log_entry:
            self.context = ChainContext.from_agent_context(context, MockMonitored("test"), "chain", budget)
            with self.context:
                runner.run(self.context, self.executor)

            print(f"\n{json.dumps(context._execution_context._executionLog.to_dict(), indent=2)}")

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
            self._execute(instance, Budget(1))

        time.sleep(1)
        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages(["first"])
        self.assertTrue(self.context.current.try_last_message.unwrap().is_error)

    def test_parallel(self):
        instance = Parallel(SkillTest("first", 0.2), SkillTest("second", 0.1))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["first", "second"])

    def test_parallel_sequence(self):
        sequence = Sequential(SkillTest("first", 0.1), SkillTest("second", 0.2))
        parallel = Parallel(sequence, SkillTest("third", 0.15), SkillTest("fourth", 0.4))
        self._execute(parallel, Budget(1))
        self.assertFalse(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["first", "second", "third", "fourth"])

    def test_parallel_sequence_with_budget(self):
        consumption = Consumption(1, "unit", "budget")
        parallel = Parallel(
            Sequential(SkillTest("first", 0.1, consumption.kind), SkillTest("second", 0.2, consumption.kind)),
            SkillTest("third", 0.15, consumption.kind),
            SkillTest("fourth", 0.4, consumption.kind),
        )
        self._execute(parallel, Budget(1, limits=[Consumption(5, "unit", "budget")]))
        self.assertFalse(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["first", "second", "third", "fourth"])
        self.assertEqual(self.context.budget._remaining[0].value, 1)

    def test_parallel_many_sequences(self):
        instance = Sequential(
            SkillTestAppend("a"),
            Parallel(
                Sequential(SkillTestAppend("b"), SkillTestAppend("c")),
                Sequential(SkillTestAppend("d"), SkillTestAppend("e")),
                Sequential(SkillTestAppend("f"), SkillTestAppend("g")),
            ),
            SkillTestMerge(["c", "e", "g"]),
        )

        self._execute(instance, Budget(1))
        self.assertEqual(self.context.last_message.message, "abcadeafg")

    def test_parallel_with_exception(self):
        instance = Parallel(SkillTest("first", 0.3), SkillTest("second", -0.2), SkillTest("third", 0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertTrue(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["third"])
        self.assertTrue(self.context.current.last_message_from_skill("second").is_error)

        time.sleep(0.2)
        self.assertSuccessMessages(["third"])

    def test_if_runner_true(self):
        instance = Sequential(If(lambda a: True, SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["maybe", "always"])

    def test_if_runner_true_with_error(self):
        instance = Sequential(If(lambda a: True, SkillTest("maybe", -0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerSkillError) as cm:
            self._execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages([])

    def test_if_runner_predicate_throw(self):
        def predicate_throw(a):
            raise Exception("predicate")

        instance = Sequential(If(lambda a: predicate_throw(a), SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerPredicateError):
            self._execute(instance, Budget(1))
        self.assertSuccessMessages([])

    def test_if_runner_false(self):
        instance = Sequential(If(lambda a: False, SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["always"])

    def test_if_runner_consume_budget(self):
        def predicate(chain_context: ChainContext) -> bool:
            chain_context.budget.add_consumption(Consumption(0.5, "unit", "budget"))
            return True

        instance = If(predicate, SkillTest("maybe", 0.1, "budget"))
        self._execute(instance, Budget(1, limits=[Consumption(2, "unit", "budget")]))
        self.assertSuccessMessages(["maybe"])
        self.assertEqual(self.context.budget._remaining[0].value, 0.5)

    def test_sequence_if_predicate_with_context_runner(self):
        def predicate(chain_context: ChainContext):
            return chain_context.current.try_last_message.unwrap().message == "first"

        instance = Sequential(SkillTest("first", 0.1), If(predicate, SkillTest("maybe", 0.2)), SkillTest("always", 0.2))
        self._execute(instance, Budget(1))
        self.assertSuccessMessages(["first", "maybe", "always"])

    def test_parallel_for(self):
        count = 100

        def generator(chain_context: ChainContext) -> Any:
            for i in range(count):
                yield i

        instance = ParallelFor(generator, SkillTest("for each", 0.01))
        self._execute(instance, Budget(2))
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
            self._execute(instance, Budget(0.5))

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
            self._execute(instance, Budget(0.5))

        self.assertIsInstance(cm.exception.__cause__, MyGeneratorError)
