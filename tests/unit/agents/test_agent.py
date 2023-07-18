import time
from unittest import TestCase

from council.agents import Agent
from council.contexts import SkillContext, ChatMessage
from council.controllers import BasicController
from council.evaluators import BasicEvaluator
from council.mocks import MockSkill
from council.runners import Budget


class TestAgent(TestCase):
    def test_execute_from_user_message(self):
        skill = MockSkill()
        message = "this is a user message"
        skill.set_action_custom_message(message)

        agent = Agent.from_skill(skill)
        result = agent.execute_from_user_message(message)
        self.assertEqual(result.best_message.message, message)

    def test_from_skill(self):
        skill = MockSkill()
        agent = Agent.from_skill(skill)

        self.assertIsInstance(agent.controller, BasicController)
        self.assertIsInstance(agent.evaluator, BasicEvaluator)
        self.assertIsInstance(agent.chains[0].runner, MockSkill)

    def test_default_budget(self):
        def action(_: SkillContext, budget: Budget) -> ChatMessage:
            return ChatMessage.skill(f"budget.deadline={budget.deadline}")

        agent = Agent.from_skill(MockSkill(action=action))
        first = agent.execute_from_user_message("first")
        time.sleep(0.01)
        second = agent.execute_from_user_message("second")

        self.assertNotEquals(first.best_message.message, second.best_message.message)
