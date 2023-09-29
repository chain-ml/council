from council.contexts import Budget, Consumption, ChainContext
from council.runners import RunnerSkillError
from council.runners.do_while_runner import DoWhile
from .helpers import RunnerTestCase, SkillTest, MySkillException


class TestWhile(RunnerTestCase):
    def test_monitors(self):
        skill = SkillTest("maybe", 0.1)
        instance = DoWhile(lambda _: True, skill)

        self.assertEqual(instance.monitor.children["doWhileBody"].name, skill.name)

    def test_while_runner_true(self):
        budget_kind = "retry"
        skill = SkillTest("skill", 0.1, budget_kind)
        instance = DoWhile(self.run_while_budget_is_not_expired, skill)

        budget = Budget(duration=10, limits=[Consumption(10, "unit", budget_kind)])
        self.execute(instance, budget)
        self.assertSuccessMessages(["skill" for _ in range(11)])

    def test_while_runner_false(self):
        skill = SkillTest("skill", 0.1)
        instance = DoWhile(lambda _: False, skill)

        budget = Budget(1)
        self.execute(instance, budget)
        self.assertSuccessMessages(["skill"])

    def test_while_runner_with_error(self):
        instance = DoWhile(lambda a: True, SkillTest("maybe", -0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages([])

    @staticmethod
    def run_while_budget_is_not_expired(context: ChainContext) -> bool:
        return not context.budget.is_expired()
