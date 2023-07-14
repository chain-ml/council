import unittest

from council.agent_tests import AgentTestCase, AgentTestCaseOutcome
from council.mocks import MockLLM, MockAgent, MockErrorAgent, MockErrorSimilarityScorer
from council.scorers import LLMSimilarityScorer


class TestAgentTestCase(unittest.TestCase):
    def test_run(self):
        llm = MockLLM.from_response("score: 10%")
        test_case = AgentTestCase("a prompt", [LLMSimilarityScorer(llm, "expected")])
        agent = MockAgent()

        result = test_case.run(agent)

        self.assertEqual(AgentTestCaseOutcome.Success, result.outcome)
        self.assertEqual(test_case.prompt, result.prompt)
        self.assertEqual(agent.message, result.actual)
        self.assertEqual(test_case.scorers[0], result.scorer_results[0].scorer)
        self.assertAlmostEqual(0.1, result.scorer_results[0].score, delta=0.0001)

    def test_run_agent_raise(self):
        class MyException(Exception):
            pass

        test_case = AgentTestCase("a prompt", [])
        agent = MockErrorAgent(exception=MyException("an error message"))

        result = test_case.run(agent)

        self.assertEqual(AgentTestCaseOutcome.Error, result.outcome)
        self.assertEqual(len(result.scorer_results), 0)
        self.assertEqual("MyException", result.error)
        self.assertEqual("an error message", result.error_message)

    def test_grader_raise(self):
        test_case = AgentTestCase("a prompt", [MockErrorSimilarityScorer()])
        agent = MockAgent()

        result = test_case.run(agent)

        self.assertEqual(test_case.prompt, result.prompt)
        self.assertEqual(AgentTestCaseOutcome.Inconclusive, result.outcome)
