import time
from typing import List
from unittest import TestCase

from council.agents import Agent
from council.chains import Chain
from council.contexts import AgentContext, Budget, ChatMessage, SkillContext
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
        def skill_execute(context: SkillContext) -> ChatMessage:
            context.logger.info("an %s log", "info")
            return ChatMessage.skill("a message")

        chains = [Chain("a chain", "do something", [MockSkill("a skill", action=skill_execute)])]
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
        self.assertEqual("an info log", result["entries"][5]["logs"][2]["message"])

    def test_plan_group(self):
        plan = [
            ExecutionUnit(Chain("a", "do something", [MockSkill()]), Budget(1)),
            ExecutionUnit(Chain("b", "do something", [MockSkill()]), Budget(1)),
            ExecutionUnit(Chain("c", "do something", [MockSkill()]), Budget(1), rank=2),
            ExecutionUnit(Chain("d", "do something", [MockSkill()]), Budget(1)),
            ExecutionUnit(Chain("e", "do something", [MockSkill()]), Budget(1), rank=4),
            ExecutionUnit(Chain("f", "do something", [MockSkill()]), Budget(1), rank=3),
            ExecutionUnit(Chain("g", "do something", [MockSkill()]), Budget(1), rank=2),
        ]

        result = Agent._group_units(plan)

        self.assertEqual(
            [["a"], ["b"], ["d"], ["c", "g"], ["f"], ["e"]], [[chain.name for chain in group] for group in result]
        )

    def test_group_execution(self):
        long = Chain(
            "long",
            "do something",
            [
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=0, message="from long chain"),
            ],
        )
        short = Chain(
            "short",
            "do something faster",
            [
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=0, message="from short chain"),
            ],
        )
        shorter = Chain(
            "shorter", "do something even faster", [MockSkill.build_wait_skill(duration=0, message="faster")]
        )

        context = AgentContext.empty(Budget(3))
        context.new_iteration()
        plan = [
            ExecutionUnit(long, context.budget, rank=1),
            ExecutionUnit(short, context.budget, rank=1),
            ExecutionUnit(shorter, context.budget, rank=1),
        ]
        agent = Agent(BasicController([long, short, shorter]), BasicEvaluator(), BasicFilter())
        agent.execute_plan(context, plan)

        self.assertFalse(context.budget.is_expired())
        self.assertLessEqual(context.budget.remaining_duration, 1)
        self.assertGreaterEqual(context.budget.remaining_duration, 0.5)
