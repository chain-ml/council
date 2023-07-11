import unittest

from council.core.execution_context import AgentMessage
from council.scorer import LLMSimilarityScorer
from council.core import ScorerException
from council.mocks import MockLLM


class TestLLMSimilarityScorer(unittest.TestCase):
    def test_parse(self):
        score = self._test_parse("score: 12.34%")
        self.assertAlmostEqual(0.1234, score, delta=0.0001)

    def test_parse_extra_spaces(self):
        score = self._test_parse("  score :  12.34 % ")
        self.assertAlmostEqual(0.1234, score, delta=0.0001)

    def test_parse_fail(self):
        with self.assertRaises(ScorerException):
            self._test_parse("this is not a score")

    @staticmethod
    def _test_parse(message: str) -> float:
        llm = MockLLM([message])
        instance = LLMSimilarityScorer(llm, "whatever")
        return instance.score(AgentMessage("does not matter", None))
