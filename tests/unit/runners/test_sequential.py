import time

from council.contexts import Budget
from council.runners import RunnerSkillError, RunnerTimeoutError, Sequential
from .helpers import MySkillException, RunnerTestCase, SkillTest


class TestSequential(RunnerTestCase):
    def test_one_skill(self):
        instance = Sequential(SkillTest("single", 0.1))
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["single"])

    def test_one_skill_timeout(self):
        instance = Sequential(SkillTest("single", 0.2))
        with self.assertRaises(RunnerTimeoutError):
            self.execute(instance, Budget(0.1))
            self.assertSuccessMessages([])
            time.sleep(0.2)
            self.assertSuccessMessages([])

    def test_sequence(self):
        instance = Sequential(SkillTest("first", 0.2), SkillTest("second", 0.1))
        self.execute(instance, Budget(0.35))
        self.assertSuccessMessages(["first", "second"])

    def test_sequence_timeout(self):
        instance = Sequential(SkillTest("first", 0.2), SkillTest("second", 0.1))
        with self.assertRaises(RunnerTimeoutError):
            self.execute(instance, Budget(0.25))
        self.assertSuccessMessages(["first"])

        time.sleep(0.1)
        self.assertSuccessMessages(["first"])

    def test_sequence_with_exception(self):
        instance = Sequential(SkillTest("first", 0.3), SkillTest("second", -0.2), SkillTest("third", 0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(1))

        time.sleep(1)
        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages(["first"])
        self.assertTrue(self.context.current.try_last_message.unwrap().is_error)
