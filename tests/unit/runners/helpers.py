import time
import unittest
from typing import List, Optional

from council.contexts import AgentContext, Budget, ChainContext, ChatMessage, SkillContext
from council.mocks import MockMonitored
from council.runners import RunnerBase, new_runner_executor
from council.skills import SkillBase


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
            context.budget.add_consumption(1, "unit", self.budget_kind)
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


class RunnerTestCase(unittest.TestCase):
    def execute(self, runner: RunnerBase, budget: Budget) -> None:
        self.executor = new_runner_executor(name="test_skill_runner")
        # print(f"\n{runner.render_as_text()}")
        print(f"\n{runner.render_as_json()}")
        context = AgentContext.empty()
        context.new_iteration()
        with context.log_entry:
            self.context = ChainContext.from_agent_context(context, MockMonitored("test"), "chain", budget)
            with self.context:
                runner.run(self.context, self.executor)

            print(f"\n{context.execution_log_to_json()}")

    def assertSuccessMessages(self, expected: List[str]):
        self.assertEqual(
            expected,
            [m.message for m in self.context.current.messages if m.is_kind_skill and m.is_ok],
        )
