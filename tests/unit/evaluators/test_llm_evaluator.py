import unittest
from typing import List

from council.evaluators import LLMEvaluator
from council.contexts import (
    AgentContext,
    Budget,
    ScoredChatMessage,
    ChatMessage,
    ChainContext,
)
from council.mocks import MockLLM, MockMonitored


class TestLLMEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.context = AgentContext.from_user_message("bla")

    def add_one_iteration_result(self):
        self.context.new_iteration()
        self.first_context = ChainContext.from_agent_context(self.context, MockMonitored(), "a chain", Budget.default())
        self.second_context = ChainContext.from_agent_context(
            self.context, MockMonitored(), "another chain", Budget.default()
        )
        self.first_context.append(ChatMessage.skill("result of a chain", source="first skill"))
        self.second_context.append(ChatMessage.skill("result of another chain", source="another skill"))

    @staticmethod
    def to_tuple_message_score(items: List[ScoredChatMessage]):
        return [(item.message.message, item.score) for item in items]

    def test_evaluate(self):
        responses = ["grade:2", "grade:10"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()

        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_chain_with_no_message(self):
        responses = ["grade:2", "grade:10"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()
        ChainContext.from_agent_context(
            self.context, MockMonitored(), "this chain does not provide any message", Budget.default()
        )

        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_fail_to_parse(self):
        response = "grade:Not a number"
        instance = LLMEvaluator(MockLLM.from_response(response))
        self.add_one_iteration_result()
        with self.assertRaises(Exception):
            instance.execute(self.context, Budget(10))

    def test_evaluate_with_execution_history(self):
        responses = ["grade:2", "grade:10"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()
        self.add_one_iteration_result()
        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))
