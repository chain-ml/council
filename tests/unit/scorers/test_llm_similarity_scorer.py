import unittest

from council.contexts import ChatMessage, ScorerContext
from council.scorers import LLMSimilarityScorer, ScorerException
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
        llm = MockLLM.from_response(message)
        instance = LLMSimilarityScorer(llm, "whatever")
        return instance.score(ScorerContext.empty(), ChatMessage.agent("does not matter"))
