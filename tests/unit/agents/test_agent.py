from typing import List
from unittest import TestCase

from council.agents import Agent
from council.chains import Chain
from council.contexts import AgentContext, ChatMessage, SkillContext
from council.controllers import BasicController, ExecutionUnit
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

    def test_initial_state(self):
        class TestController(BasicController):
            def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
                return [
                    ExecutionUnit(chain, budget, ChatMessage.chain(f"from {chain.name}", source=chain.name))
                    for chain in chains
                ]

        def skill_action(context: SkillContext, budget: Budget) -> ChatMessage:
            message = context.try_last_message.unwrap()
            return ChatMessage.skill(message.message)

        skill = MockSkill(action=skill_action)

        agent = Agent(TestController(), [Chain("a chain", description="", runners=[skill])], BasicEvaluator())
        result = agent.execute_from_user_message("run")
        self.assertEqual(result.best_message.message, "from a chain")
