import unittest

import dotenv

from council.contexts import AgentMessage
from council.llm import AzureLLM
from council.scorers import LLMSimilarityScorer


class TestLLMSimilarityScorer(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

    def test_similarity_hello(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
            """
        instance = LLMSimilarityScorer(self.llm, expected)

        score = instance.score(AgentMessage(expected, None))
        self.assertAlmostEqual(1.0, score, delta=0.1)

    def test_similarity_wrong_assistant(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
            """
        instance = LLMSimilarityScorer(self.llm, expected)
        actual = """
            I am an assistant export in writing Python code.
            How can I help you today?
        """
        score = instance.score(AgentMessage(actual, None))
        self.assertAlmostEqual(0.5, score, delta=0.1)

    def test_similarity_support_agent(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
            """
        instance = LLMSimilarityScorer(self.llm, expected)
        actual = """
            Hi, I'm here to assist you in your support case.
            How can I help you today?
            """
        score = instance.score(AgentMessage(actual, None))
        self.assertAlmostEqual(0.33, score, delta=0.1)

    def test_similarity_unrelated(self):
        expected = """
            I am an assistant expert in build SQL query.
            How can I help you today?
            """
        instance = LLMSimilarityScorer(self.llm, expected)

        score = instance.score(AgentMessage("the capital of France is Paris", None))
        self.assertAlmostEqual(0.0, score, delta=0.1)
