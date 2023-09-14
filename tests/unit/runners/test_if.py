from council.contexts import Budget, ChainContext, Consumption
from council.runners import If, RunnerPredicateError, RunnerSkillError, Sequential
from .helpers import MySkillException, RunnerTestCase, SkillTest


class TestIf(RunnerTestCase):
    def test_monitors(self):
        then_skill = SkillTest("maybe", 0.1)
        instance = If(lambda a: True, then_skill)

        self.assertEqual(instance.monitor.children["then"].name, then_skill.name)
        self.assertIsNone(instance.monitor.children.get("else"))

    def test_monitors_with_else(self):
        then_skill = SkillTest("then skill", 0.1)
        else_skill = SkillTest("else skill", 0.1)
        instance = If(lambda a: True, then_skill, else_skill)

        self.assertEqual(instance.monitor.children["then"].name, then_skill.name)
        self.assertEqual(instance.monitor.children["else"].name, else_skill.name)

    def test_if_runner_true(self):
        instance = Sequential(
            If(lambda a: True, SkillTest("then", 0.1), SkillTest("else", 0.1)), SkillTest("always", 0.2)
        )
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["then", "always"])

    def test_if_runner_false_with_else(self):
        instance = Sequential(
            If(lambda a: False, SkillTest("then", 0.1), SkillTest("else", 0.1)), SkillTest("always", 0.2)
        )
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["else", "always"])

    def test_if_runner_false_without_else(self):
        instance = Sequential(If(lambda a: False, SkillTest("then", 0.1)), SkillTest("always", 0.2))
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["always"])

    def test_if_runner_true_with_error(self):
        instance = Sequential(If(lambda a: True, SkillTest("maybe", -0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages([])

    def test_if_runner_predicate_throw(self):
        def predicate_throw(a):
            raise Exception("predicate")

        instance = Sequential(If(lambda a: predicate_throw(a), SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        with self.assertRaises(RunnerPredicateError):
            self.execute(instance, Budget(1))
        self.assertSuccessMessages([])

    def test_if_runner_false(self):
        instance = Sequential(If(lambda a: False, SkillTest("maybe", 0.1)), SkillTest("always", 0.2))
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["always"])

    def test_if_runner_consume_budget(self):
        def predicate(chain_context: ChainContext) -> bool:
            chain_context.budget.add_consumption(0.5, "unit", "budget")
            return True

        instance = If(predicate, SkillTest("maybe", 0.1, "budget"))
        self.execute(instance, Budget(1, limits=[Consumption(2, "unit", "budget")]))
        self.assertSuccessMessages(["maybe"])
        self.assertEqual(self.context.budget._remaining[0].value, 0.5)

    def test_sequence_if_predicate_with_context_runner(self):
        def predicate(chain_context: ChainContext):
            return chain_context.current.try_last_message.unwrap().message == "first"

        instance = Sequential(SkillTest("first", 0.1), If(predicate, SkillTest("maybe", 0.2)), SkillTest("always", 0.2))
        self.execute(instance, Budget(1))
        self.assertSuccessMessages(["first", "maybe", "always"])
