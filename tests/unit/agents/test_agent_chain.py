import unittest

from council.agents import Agent, AgentChain
from council.chains import Chain
from council.contexts import Budget
from council.controllers import BasicController
from council.evaluators import BasicEvaluator
from council.filters import BasicFilter
from council.mocks import MockSkill


class TestAgentChain(unittest.TestCase):
    def setUp(self) -> None:
        mock_skill = MockSkill("a skill")
        mock_skill.set_action_custom_message("hi from skill")
        chains = [Chain("a chain", "do something", [mock_skill])]
        inner_agent = Agent(BasicController(chains), BasicEvaluator(), BasicFilter(), name="inner agent")

        agent_chain = AgentChain("agent chain", "", inner_agent)
        self.agent = Agent(BasicController([agent_chain]), BasicEvaluator(), BasicFilter(), name="outer agent")

    def test_agent_chain_monitor(self):
        self.assertEqual(self.agent.monitor.children["chains[0]"].name, "agent chain")
        self.assertEqual(self.agent.monitor.children["chains[0]"].children["agent"].name, "inner agent")
        self.assertIsNone(self.agent.monitor.children["chains[0]"].children.get("runner"))

    def test_run(self):
        result = self.agent.execute_from_user_message("hi", Budget(1))
        self.assertEqual("hi from skill", result.best_message.message)
        self.assertEqual("a skill", result.best_message.source)
