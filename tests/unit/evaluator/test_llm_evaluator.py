import unittest
from typing import List

from council.core import Budget
from council.evaluator import LLMEvaluator
from council.core.execution_context import (
    AgentContext,
    ChatHistory,
    SkillSuccessMessage,
    ScoredAgentMessage,
    AgentMessage,
)
from council.mocks import MockLLM


class TestLLMEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.context = AgentContext(ChatHistory())
        self.context.chatHistory.add_user_message("bla")

    def add_one_iteration_result(self):
        self.first_context = self.context.new_chain_context("a chain")
        self.second_context = self.context.new_chain_context("another chain")
        self.first_context.current.messages.append(SkillSuccessMessage("first skill", "result of a chain"))
        self.second_context.current.messages.append(SkillSuccessMessage("another skill", "result of another chain"))

    @staticmethod
    def to_tuple_message_score(items: List[ScoredAgentMessage]):
        return [(item.message.message, item.score) for item in items]

    def test_evaluate(self):
        responses = ["result of a chain:2", "result of another chain:10"]
        expected = [
            ScoredAgentMessage(AgentMessage("result of a chain", None), 2),
            ScoredAgentMessage(AgentMessage("result of another chain", None), 10),
        ]

        self.add_one_iteration_result()

        result = LLMEvaluator(MockLLM(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_chain_with_no_message(self):
        responses = ["result of a chain:2", "result of another chain:10"]
        expected = [
            ScoredAgentMessage(AgentMessage("result of a chain", None), 2),
            ScoredAgentMessage(AgentMessage("result of another chain", None), 10),
        ]

        self.add_one_iteration_result()
        self.context.new_chain_context("this chain does not provide any message")

        result = LLMEvaluator(MockLLM(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_fail_to_parse(self):
        responses = ["a response with no score"]
        instance = LLMEvaluator(MockLLM(responses))
        self.add_one_iteration_result()
        with self.assertRaises(Exception):
            instance.execute(self.context, Budget(10))

    def test_evaluate_with_execution_history(self):
        responses = ["result of a chain:2", "result of another chain:10"]
        expected = [
            ScoredAgentMessage(AgentMessage("result of a chain", None), 2),
            ScoredAgentMessage(AgentMessage("result of another chain", None), 10),
        ]

        self.add_one_iteration_result()
        self.add_one_iteration_result()
        result = LLMEvaluator(MockLLM(responses)).execute(self.context, Budget(10))
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))
