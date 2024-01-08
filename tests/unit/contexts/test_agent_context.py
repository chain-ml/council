import unittest

from council import Agent, AgentContext, Budget
from council.mocks import MockSkill


class TestAgentContext(unittest.TestCase):
    def test_log_timeout(self):
        skill = MockSkill.build_wait_skill(duration=1)

        context = AgentContext.empty(budget=Budget(duration=0.5))
        agent = Agent.from_skill(skill)
        agent.execute(context)

        result = context.execution_log_to_json()
        self.assertIn('"error": "RunnerTimeoutError: MockSkill"', result)
