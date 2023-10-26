import unittest
from typing import List

from council.contexts import (
    AgentContext,
    Budget,
    ScoredChatMessage,
)
from council.filters import FilterException
from council.filters.llm_filter import LLMFilter
from council.mocks import MockLLM
from unit.filters import build_scored_message


class TestLLMFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.context = AgentContext.from_user_message("bla", Budget(10))

    def add_one_iteration_result(self):
        self.context.new_iteration()
        messages = [
            build_scored_message("result of a chain", 8),
            build_scored_message("result of another chain", 2),
        ]
        self.context.set_evaluation(messages)

    @staticmethod
    def to_tuple_message_score(items: List[ScoredChatMessage]):
        return [(item.message.message, item.score) for item in items]

    def test_filter(self):
        responses = [
            "is_filtered:True<->index:1<->justification:None",
            "is_filtered:False<->index:2<->justification:None",
        ]
        expected = [build_scored_message("result of another chain", 2)]

        self.add_one_iteration_result()

        result = LLMFilter(MockLLM.from_multi_line_response(responses)).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_filter_fail(self):
        responses = [
            "is_filtered:NotABoolean<->index:1<->justification:None",
            "is_filtered:False<->index:2<->justification:None",
        ]

        self.add_one_iteration_result()

        with self.assertRaises(FilterException):
            _ = LLMFilter(MockLLM.from_multi_line_response(responses)).execute(self.context)
