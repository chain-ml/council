import json
import unittest

from council.agent_tests import AgentTestSuite
from council.scorer import LLMSimilarityScorer
from council.mocks import MockLLM, MockAgent


class TestAgentTestSuite(unittest.TestCase):
    def test_run_tests(self):
        suite = (
            AgentTestSuite()
            .add_test_case(
                prompt="what is the weather today",
                scorers=[
                    LLMSimilarityScorer(MockLLM(["score: 55%"]), "I cannot predict the weather"),
                    LLMSimilarityScorer(MockLLM(["score: 100%"]), "weather is great"),
                ],
            )
            .add_test_case(
                prompt="introduce yourself",
                scorers=[
                    LLMSimilarityScorer(MockLLM(["score: 95%"]), "I'm an agent specialized in predicting the weather"),
                    LLMSimilarityScorer(MockLLM(["score: 10%"]), "sorry, I don't understand"),
                ],
            )
        )

        agent = MockAgent(message="agent message", score=1.0, sleep=0.2, sleep_interval=0.1)

        result = suite.run(agent, show_progressbar=False)
        result_dict = result.to_dict()

        # test it can serialize
        print(json.dumps(result_dict, indent=2))

        self.assertEqual(2, len(result_dict["results"]), "test cases")
