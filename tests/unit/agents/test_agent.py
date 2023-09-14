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
        chain = agent.controller.chains[0]
        self.assertIsInstance(chain, Chain)
        self.assertIsInstance(chain.runner, MockSkill)

    def test_default_budget(self):
        def action(context: SkillContext) -> ChatMessage:
            return ChatMessage.skill(f"budget.deadline={context.budget.deadline}")

        agent = Agent.from_skill(MockSkill(action=action))
        first = agent.execute_from_user_message("first")
        time.sleep(0.01)
        second = agent.execute_from_user_message("second")

        self.assertNotEquals(first.best_message.message, second.best_message.message)

    def test_initial_state(self):
        class TestController(BasicController):
            def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
                return [
                    ExecutionUnit(chain, context.budget, ChatMessage.chain(f"from {chain.name}", source="controller"))
                    for chain in self._chains
                ]

        def skill_action(context: SkillContext) -> ChatMessage:
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
            def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
                return [
                    ExecutionUnit(
                        chain,
                        context.budget,
                        ChatMessage.chain(f"from {chain.name} {index}", source=chain.name),
                        name=f"{chain.name}[{index}]",
                    )
                    for chain in self._chains
                    for index in [0, 1, 2]
                ]

        def skill_action(context: SkillContext) -> ChatMessage:
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

    def test_agent_graph(self):
        chains = [Chain("a chain", "do something", [MockSkill("a skill")])]
        agent = Agent(BasicController(chains), BasicEvaluator(), BasicFilter(), name="an agent")
        result = agent.render_as_dict()
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "Agent")
        self.assertEqual(result["properties"]["name"], "an agent")
        self.assertEqual(result["children"][0]["value"]["type"], "BasicController")
        self.assertEqual(result["children"][1]["value"]["type"], "Chain")
        self.assertEqual(result["children"][1]["value"]["children"][0]["value"]["type"], "MockSkill")
        self.assertEqual(result["children"][2]["value"]["type"], "BasicEvaluator")
        self.assertEqual(result["children"][3]["value"]["type"], "BasicFilter")

        print(agent.render_as_json())

    def test_agent_log(self):
        chains = [Chain("a chain", "do something", [MockSkill("a skill")])]
        agent = Agent(BasicController(chains), BasicEvaluator(), BasicFilter(), name="an agent")

        context = AgentContext.from_user_message("run")
        agent.execute(context)
        result = context.execution_log_to_dict()

        self.assertIsNotNone(result)
        expected_sources = [
            "agent",
            "agent/iterations[0]",
            "agent/iterations[0]/controller",
            "agent/iterations[0]/execution(a chain)",
            "agent/iterations[0]/execution(a chain)/chain(a chain)",
            "agent/iterations[0]/execution(a chain)/chain(a chain)/runner",
            "agent/iterations[0]/evaluator",
            "agent/iterations[0]/filter",
        ]

        self.assertEqual(expected_sources, [item["source"] for item in result["entries"]])
