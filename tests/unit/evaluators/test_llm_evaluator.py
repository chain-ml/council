import unittest
from typing import List

from council.evaluators import EvaluatorException, LLMEvaluator
from council.contexts import (
    AgentContext,
    Budget,
    ScoredChatMessage,
    ChatMessage,
    ChainContext,
)
from council.mocks import MockLLM, MockMonitored, MockMultipleResponses


class TestLLMEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.context = AgentContext.from_user_message("bla", Budget(10))

    def add_one_iteration_result(self):
        self.context.new_iteration()
        self.first_context = ChainContext.from_agent_context(self.context, MockMonitored(), "a chain")
        self.second_context = ChainContext.from_agent_context(self.context, MockMonitored(), "another chain")
        self.first_context.append(ChatMessage.skill("result of a chain", source="first skill"))
        self.second_context.append(ChatMessage.skill("result of another chain", source="another skill"))

    @staticmethod
    def to_tuple_message_score(items: List[ScoredChatMessage]):
        return [(item.message.message, item.score) for item in items]

    @staticmethod
    def to_score(items: List[ScoredChatMessage]):
        return [item.score for item in items]

    def test_evaluate(self):
        responses = ["grade:2<->index:1<->justification:None", "grade:10<->index:2<->justification:None"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()

        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_chain_with_no_message(self):
        responses = ["index:1<->grade:2<->justification:None", "grade:10<->index:2<->justification:None"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()
        ChainContext.from_agent_context(self.context, MockMonitored(), "this chain does not provide any message")

        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_evaluate_fail_to_parse_first_2_answers(self):
        llm_responses = [
            ["grade:NotANumber<->index:1<->justification:None", "grade:6<->index:2<->justification:None"],
            ["grade:6<->index:2<->justification:None", "grade:3.<->index:NotAnInteger<->justification:None"],
            ["grade:4<->index:1<->justification:Because", "grade:8<->index:2<->justification:None"],
        ]

        instance = LLMEvaluator(llm=MockLLM(action=MockMultipleResponses(responses=llm_responses).call))
        self.add_one_iteration_result()
        result = instance.execute(self.context)
        self.assertEqual(self.to_score(result), [4.0, 8.0])

    def test_evaluate_fail(self):
        llm_responses = [
            ["grade:NotANumber<->index:1<->justification:None", "grade:6<->index:2<->justification:None"],
            ["grade:6<->index:2<->justification:None", "grade:3.<->index:NotAnInteger<->justification:None"],
            ["grade:4<->justification:Because", "grade:8<->index:2<->justification:None"],
        ]

        instance = LLMEvaluator(llm=MockLLM(action=MockMultipleResponses(responses=llm_responses).call))
        self.add_one_iteration_result()
        with self.assertRaises(EvaluatorException):
            instance.execute(self.context)

    def test_evaluate_with_execution_history(self):
        responses = ["grade:2<->index:1<->justification:None", "grade:10<->index:2<->justification:None"]
        expected = [
            ScoredChatMessage(ChatMessage.agent("result of a chain"), 2),
            ScoredChatMessage(ChatMessage.agent("result of another chain"), 10),
        ]

        self.add_one_iteration_result()
        self.add_one_iteration_result()
        result = LLMEvaluator(MockLLM.from_multi_line_response(responses)).execute(self.context)
        self.assertEqual(self.to_tuple_message_score(expected), self.to_tuple_message_score(result))

    def test_monitors(self):
        llm = MockLLM()
        instance = LLMEvaluator(llm)

        self.assertEqual(instance.monitor.children["llm"].type, "MockLLM")
