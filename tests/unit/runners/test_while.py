from council.contexts import ChainContext, Budget, Consumption
from council.runners import While, RunnerSkillError, RunnerPredicateError
from tests.unit.runners.helpers import RunnerTestCase, SkillTest, MySkillException

BUDGET_KIND: str = "iteration"


class TestWhile(RunnerTestCase):
    def test_monitors(self):
        skill = SkillTest("maybe", 0.1)
        instance = While(lambda _: True, skill)

        self.assertEqual(instance.monitor.children["whileBody"].name, skill.name)

    def test_while_runner_has_budget(self):
        skill = SkillTest("skill", 0.1, BUDGET_KIND)
        instance = While(self.run_while_budget_is_not_expired, skill)

        max_iteration = 10
        budget = Budget(duration=100, limits=[Consumption(max_iteration, "unit", BUDGET_KIND)])
        self.execute(instance, budget)
        self.assertSuccessMessages(["skill" for _ in range(max_iteration)])

    def test_while_runner_false(self):
        skill = SkillTest("skill", 0.1)
        instance = While(lambda _: False, skill)

        self.execute(instance, Budget(10))
        self.assertSuccessMessages([])

    def test_while_runner_with_error(self):
        instance = While(lambda _: True, SkillTest("skill", -0.1))
        with self.assertRaises(RunnerSkillError) as cm:
            self.execute(instance, Budget(1))

        self.assertIsInstance(cm.exception.__cause__, MySkillException)
        self.assertSuccessMessages([])

    def test_while_runner_with_predicate_exception(self):
        instance = While(self.predicate_raising_exception, SkillTest("skill", 0.1))
        with self.assertRaises(RunnerPredicateError):
            self.execute(instance, Budget(1))

        self.assertSuccessMessages([])

    @staticmethod
    def run_while_budget_is_not_expired(context: ChainContext) -> bool:
        return context.budget.can_consume(1.0, "unit", BUDGET_KIND)

    @staticmethod
    def predicate_raising_exception(context: ChainContext) -> bool:
        raise Exception("the predicate function raised an exception")
