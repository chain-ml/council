import time
from typing import List
from unittest import TestCase

from council.agents import Agent
from council.chains import Chain
from council.contexts import AgentContext, ChatMessage, SkillContext
from council.controllers import BasicController, ExecutionUnit
from council.evaluators import BasicEvaluator
from council.filters import BasicFilter
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
        self.assertIsInstance(agent.controller.chains[0].runner, MockSkill)

    def test_default_budget(self):
        def action(_: SkillContext, budget: Budget) -> ChatMessage:
            return ChatMessage.skill(f"budget.deadline={budget.deadline}")

        agent = Agent.from_skill(MockSkill(action=action))
        first = agent.execute_from_user_message("first")
        time.sleep(0.01)
        second = agent.execute_from_user_message("second")

        self.assertNotEquals(first.best_message.message, second.best_message.message)

    def test_initial_state(self):
        class TestController(BasicController):
            def _execute(self, context: AgentContext, budget: Budget) -> List[ExecutionUnit]:
                return [
                    ExecutionUnit(chain, budget, ChatMessage.chain(f"from {chain.name}", source=chain.name))
                    for chain in self._chains
                ]

        def skill_action(context: SkillContext, budget: Budget) -> ChatMessage:
            message = context.try_last_message.unwrap()
            return ChatMessage.skill(message.message)

        skill = MockSkill(action=skill_action)

        agent = Agent(
            TestController([Chain("a chain", description="", runners=[skill])]), BasicEvaluator(), BasicFilter()
        )
        result = agent.execute_from_user_message("run")
        self.assertEqual(result.best_message.message, "from a chain")

    def test_run_multiple_instances_of_a_chain(self):
        class TestController(BasicController):
            def _execute(self, context: AgentContext, budget: Budget) -> List[ExecutionUnit]:
                return [
                    ExecutionUnit(
                        chain,
                        budget,
                        ChatMessage.chain(f"from {chain.name} {index}", source=chain.name),
                        name=f"{chain.name}[{index}]",
                    )
                    for chain in self._chains
                    for index in [0, 1, 2]
                ]

        def skill_action(context: SkillContext, budget: Budget) -> ChatMessage:
            message = context.try_last_message.unwrap()
            return ChatMessage.skill(message.message)

        skill = MockSkill(action=skill_action)
        chain_one = Chain("chain one", description="", runners=[skill])
        chain_two = Chain("chain two", description="", runners=[skill])

        agent = Agent(TestController([chain_one, chain_two]), BasicEvaluator(), BasicFilter())
        result = agent.execute_from_user_message("run")
        self.assertEqual(
            [
                "from chain one 0",
                "from chain one 1",
                "from chain one 2",
                "from chain two 0",
                "from chain two 1",
                "from chain two 2",
            ],
            [item.message.message for item in result.messages],
        )
