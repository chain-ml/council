import unittest

from council.chains import Chain
from council.controllers import LLMController
from council.contexts import AgentContext, Budget
from council.mocks import MockLLM


class LLMControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chains = [Chain("first", "", []), Chain("second", "", []), Chain("third", "", [])]
        self.context = AgentContext.from_user_message("bla")

    def test_plan_parse(self):
        llm = MockLLM.from_multi_line_response(["name: first;score: 10;because", "name: second;score: 6;because"])
        controller = LLMController(chains=self.chains, llm=llm)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first", "second"], [item.chain.name for item in result])

    def test_plan_parse_top_1(self):
        llm = MockLLM.from_multi_line_response(["name: first;score: 10;because", "name: second;score: 6;because"])
        controller = LLMController(chains=self.chains, llm=llm, top_k=1)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_top_1_unsorted(self):
        llm = MockLLM.from_multi_line_response(["name: second;6;", "name: first;10;"])
        controller = LLMController(chains=self.chains, llm=llm, top_k=1)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_response_threshold(self):
        llm = MockLLM.from_multi_line_response(["name: second;6;", "name: first;10;"])
        controller = LLMController(chains=self.chains, llm=llm, response_threshold=7.0)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_case_mismatch(self):
        chains = [Chain("first", "", []), Chain("second", "", []), Chain("ThiRd", "", [])]
        llm = MockLLM.from_multi_line_response(["name: FiRsT;6;", "name: second;4;", "name: third;5;"])
        controller = LLMController(chains=chains, llm=llm, top_k=5)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first", "ThiRd", "second"], [item.chain.name for item in result])

    def test_plan_parse_no_matching_chain(self):
        llm = MockLLM.from_multi_line_response(["name: first;10;", "name: secondDoesNotExists;4;", "name: third;2;"])
        controller = LLMController(chains=self.chains, llm=llm, top_k=3)
        result = controller.execute(self.context, Budget(10))
        self.assertEqual(["first", "third"], [item.chain.name for item in result])
