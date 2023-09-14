import time

from council.contexts import Budget, Consumption
from council.runners import Parallel, RunnerSkillError, Sequential
from .helpers import MySkillException, RunnerTestCase, SkillTest, SkillTestAppend, SkillTestMerge


class TestParallel(RunnerTestCase):
    def test_parallel(self):
        instance = Parallel(SkillTest("first", 0.2), SkillTest("second", 0.1))
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["first", "second"])

    def test_parallel_sequence(self):
        sequence = Sequential(SkillTest("first", 0.1), SkillTest("second", 0.2))
        parallel = Parallel(sequence, SkillTest("third", 0.15), SkillTest("fourth", 0.4))
        self.execute(parallel, Budget(1))
        self.assertFalse(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["first", "second", "third", "fourth"])

    def test_parallel_sequence_with_budget(self):
        consumption = Consumption(1, "unit", "budget")
        parallel = Parallel(
            Sequential(SkillTest("first", 0.1, consumption.kind), SkillTest("second", 0.2, consumption.kind)),
            SkillTest("third", 0.15, consumption.kind),
            SkillTest("fourth", 0.4, consumption.kind),
        )
        self.execute(parallel, Budget(1, limits=[Consumption(5, "unit", "budget")]))
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

        self.execute(instance, Budget(1))
        self.assertEqual(self.context.last_message.message, "abcadeafg")

    def test_parallel_with_exception(self):
        instance = Parallel(SkillTest("first", 0.3), SkillTest("second", -0.2), SkillTest("third", 0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertTrue(self.context.cancellation_token.cancelled)
        self.assertSuccessMessages(["third"])
        self.assertTrue(self.context.current.last_message_from_skill("second").is_error)

        time.sleep(0.2)
        self.assertSuccessMessages(["third"])
