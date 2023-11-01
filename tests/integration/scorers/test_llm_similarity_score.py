import unittest

import dotenv

from council.contexts import ChatMessage, ScorerContext
from council.scorers import LLMSimilarityScorer

from .. import get_test_default_llm


class TestLLMSimilarityScorer(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = get_test_default_llm()

    def test_similarity_hello(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
            """
        instance = LLMSimilarityScorer(self.llm, expected)

        score = instance.score(ScorerContext.empty(), ChatMessage.agent(expected))
        self.assertAlmostEqual(1.0, score, delta=0.1)

    def test_similarity_wrong_assistant(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
        """
        instance = LLMSimilarityScorer(self.llm, expected)
        actual = """
            I am an assistant expert in writing Python code.
            How can I help you today?
        """
        score = instance.score(ScorerContext.empty(), ChatMessage.agent(actual))
        self.assertAlmostEqual(0.5, score, delta=0.1)

    def test_similarity_support_agent(self):
        expected = """
            I am an expert in build SQL query.
            How can I help you today?
        """
        instance = LLMSimilarityScorer(self.llm, expected)
        actual = """
            Hi, I'm here to assist you in your support case.
            How can I help you today?
        """
        score = instance.score(ScorerContext.empty(), ChatMessage.agent(actual))
        self.assertAlmostEqual(0.25, score, delta=0.1)

    def test_similarity_unrelated(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
        """
        instance = LLMSimilarityScorer(self.llm, expected)

        score = instance.score(ScorerContext.empty(), ChatMessage.agent("the capital of France is Paris"))
        self.assertAlmostEqual(0.0, score, delta=0.1)
