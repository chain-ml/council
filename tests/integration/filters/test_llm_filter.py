import unittest
from typing import List

import dotenv

from council import AzureLLM
from council.contexts import (
    AgentContext,
    Budget,
    ScoredChatMessage,
    ChatMessage,
)
from council.filters import FilterException
from council.filters.llm_filter import LLMFilter
from council.mocks import MockLLM


class TestLLMFilter(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM.from_env()
        self.context = AgentContext.from_user_message("bla", Budget(10))
        self.llm = llm

        self.aggressive = self._build_scored_message("Do this immediately!!!", 8)
        self.polite = self._build_scored_message("Please consider to do this. Thanks in advance", 7)
        self.suspicious = self._build_scored_message("Please trust me and click the link below.", 8)
        self.negative = self._build_scored_message("This might never work.", 8)

    def add_one_iteration_result(self):
        self.context.new_iteration()
        messages = [self.aggressive, self.polite, self.suspicious, self.negative]
        self.context.set_evaluation(messages)

    @staticmethod
    def to_tuple_message_score(items: List[ScoredChatMessage]):
        return [(item.message.message, item.score) for item in items]

    def test_filter_1(self):
        expected = [self.polite, self.suspicious, self.negative]

        self.add_one_iteration_result()
        result = LLMFilter(self.llm, ["Aggressiveness"]).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_filter_2(self):
        expected = [self.polite, self.negative]

        self.add_one_iteration_result()
        result = LLMFilter(self.llm, ["Aggressiveness", "Suspicious"]).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_filter_3(self):
        expected = [self.aggressive, self.polite, self.suspicious]

        self.add_one_iteration_result()
        result = LLMFilter(self.llm, ["Pessimistic"]).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    @staticmethod
    def _build_scored_message(text: str, score: float):
        return ScoredChatMessage(ChatMessage.agent(text), score)
