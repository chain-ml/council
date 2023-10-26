import unittest
from typing import List

from council import BasicFilter
from council.contexts import (
    AgentContext,
    Budget,
    ScoredChatMessage,
)
from . import build_scored_message


class TestLLMFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.context = AgentContext.from_user_message("bla", Budget(10))

    def add_one_iteration_result(self):
        self.context.new_iteration()
        messages = [
            build_scored_message("result of a chain", 8),
            build_scored_message("result of another chain", 2),
            build_scored_message("result of another chain", 10),
        ]
        self.context.set_evaluation(messages)

    @staticmethod
    def to_tuple_message_score(items: List[ScoredChatMessage]):
        return [(item.message.message, item.score) for item in items]

    def test_filter_top_k(self):
        expected = [
            build_scored_message("result of a chain", 8),
        ]

        self.add_one_iteration_result()

        result = BasicFilter(top_k=1).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_filter_threshold(self):
        expected = [build_scored_message("result of a chain", 8), build_scored_message("result of another chain", 10)]

        self.add_one_iteration_result()

        result = BasicFilter(score_threshold=5).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))
