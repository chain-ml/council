import unittest

from council.chains import Chain
from council.controllers import LLMController
from council.contexts import AgentContext, ChatHistory
from council.mocks import MockLLM
from council.runners import Budget


class LLMControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chains = [Chain("first", "", []), Chain("second", "", []), Chain("third", "", [])]
        history = ChatHistory()
        history.add_user_message("bla")
        self.context = AgentContext(history)

    def test_plan_parse(self):
        llm = MockLLM.from_multi_line_response(["name: first;score: 10;because", "name: second;score: 6;because"])
        controller = LLMController(llm)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first", "second"], [item.chain.name for item in result])

    def test_plan_parse_top_1(self):
        llm = MockLLM.from_multi_line_response(["name: first;score: 10;because", "name: second;score: 6;because"])
        controller = LLMController(llm, top_k_execution_plan=1)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_top_1_unsorted(self):
        llm = MockLLM.from_multi_line_response(["name: second;6;", "name: first;10;"])
        controller = LLMController(llm, top_k_execution_plan=1)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_case_mismatch(self):
        chains = [Chain("first", "", []), Chain("second", "", []), Chain("ThiRd", "", [])]
        llm = MockLLM.from_multi_line_response(["name: FiRsT;6;", "name: second;4;", "name: third;5;"])
        controller = LLMController(llm, top_k_execution_plan=5)
        result = controller.get_plan(self.context, chains, Budget(10))
        self.assertEqual(["first", "ThiRd", "second"], [item.chain.name for item in result])

    def test_plan_parse_no_matching_chain(self):
        llm = MockLLM.from_multi_line_response(["name: first;10;", "name: secondDoesNotExists;4;", "name: third;2;"])
        controller = LLMController(llm, top_k_execution_plan=3)
        result = controller.get_plan(self.context, self.chains, Budget(10))
        self.assertEqual(["first", "third"], [item.chain.name for item in result])
