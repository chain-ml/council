import unittest
from typing import List, Any

from council.chains import Chain
from council.controllers import LLMController
from council.contexts import AgentContext, ChatHistory
from council.llm import LLMMessage, LLMBase
from council.mocks import MockLLM
from council.runners import Budget


class LLMControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chains = [Chain("first", "", []), Chain("second", "", []), Chain("third", "", [])]
        history = ChatHistory()
        history.add_user_message("bla")
        self.context = AgentContext(history)

    def test_plan_parse(self):
        llm = MockLLM.from_multi_line_response(["first;10", "second;6"])
        controller = LLMController(llm)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first", "second"], [item.chain.name for item in result])

    def test_plan_parse_top_1(self):
        llm = MockLLM.from_multi_line_response(["first;10", "second;6"])
        controller = LLMController(llm, top_k_execution_plan=1)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_top_1_unsorted(self):
        llm = MockLLM.from_multi_line_response(["second;6", "first;10"])
        controller = LLMController(llm, top_k_execution_plan=1)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_case_mismatch(self):
        chains = [Chain("first", "", []), Chain("second", "", []), Chain("ThiRd", "", [])]
        llm = MockLLM.from_multi_line_response(["FiRsT;6", "second;4", "third;5"])
        controller = LLMController(llm, top_k_execution_plan=5)
        result = controller.get_plan(self.context, chains, Budget(10))
        self.assertEqual(["first", "ThiRd", "second"], [item.chain.name for item in result])

    def test_plan_parse_no_matching_chain(self):
        llm = MockLLM.from_multi_line_response(["first;10", "secondDoesNotExists;4", "third;2"])
        controller = LLMController(llm, top_k_execution_plan=3)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first", "third"], [item.chain.name for item in result])
