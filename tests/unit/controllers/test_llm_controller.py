import unittest

from council.agents import Agent
from council.chains import Chain
from council.controllers import LLMController, ControllerException
from council.contexts import AgentContext, Budget
from council.evaluators import BasicEvaluator
from council.filters import BasicFilter
from council.mocks import MockLLM, MockSkill, MockMultipleResponses


class LLMControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chains = [Chain("first", "", []), Chain("second", "", []), Chain("third", "", [])]
        self.context = AgentContext.from_user_message("bla", Budget(10))

    def test_plan_parse(self):
        llm = MockLLM.from_multi_line_response(
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "score: 4<->name: third<->instructions: None<->justification: because",
                "name: second<->score: 6<->instructions: None<->justification: because",
            ]
        )
        controller = LLMController(chains=self.chains, llm=llm)
        result = controller.execute(self.context)
        self.assertEqual(["first", "second", "third"], [item.chain.name for item in result])

    def test_plan_parse_top_1(self):
        llm = MockLLM.from_multi_line_response(
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
            ]
        )
        controller = LLMController(chains=self.chains, llm=llm, top_k=1)
        result = controller.execute(self.context)
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_response_threshold(self):
        llm = MockLLM.from_multi_line_response(
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "name: second<->score: 6<->instructions: None<->justification: because",
                "name: third<->score: 6<->instructions: None<->justification: because",
            ]
        )
        controller = LLMController(chains=self.chains, llm=llm, response_threshold=7)
        result = controller.execute(self.context)
        self.assertEqual(["first"], [item.chain.name for item in result])

    def test_plan_parse_case_mismatch(self):
        chains = [Chain("first", "", []), Chain("second", "", []), Chain("ThiRd", "", [])]
        llm = MockLLM.from_multi_line_response(
            [
                "name: FirsT<->score: 10<->instructions: None<->justification: because",
                "name: secOnd<->score: 6<->instructions: None<->justification: because",
                "name: thIrd<->justification: because<->score: 8<->instructions: None",
            ]
        )
        controller = LLMController(chains=chains, llm=llm, top_k=5)
        result = controller.execute(self.context)
        self.assertEqual(["first", "ThiRd", "second"], [item.chain.name for item in result])

    def test_plan_retry_until_success(self):
        llm_responses = [
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "name: second<->instructions: None<->justification: Missing Score",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "name: secondDoesNotExists<->score: 6<->instructions: None<->justification: Missing chain",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
            [
                "name: first<->score: 4<->instructions: None<->justification: because",
                "name: second<->score: 6<->instructions: None<->justification: because",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
        ]

        llm = MockLLM(action=MockMultipleResponses(responses=llm_responses))

        controller = LLMController(chains=self.chains, llm=llm, top_k=3)
        result = controller.execute(self.context)
        self.assertEqual(["second", "first", "third"], [item.chain.name for item in result])

    def test_plan_fail(self):
        invalid_llm_responses = [
            [
                "name: first<->score: 10<->instructions: None<->justification: Missing scores",
            ],
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "name: secondDoesNotExists<->score: 6<->instructions: None<->justification: Invalid Chain name",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
            [
                "name: first<->score: 4<->instructions: None<->justification: because",
                "name: second<->score: A<->instructions: None<->justification: Invalid Score",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
        ]

        llm = MockLLM(action=MockMultipleResponses(responses=invalid_llm_responses))

        controller = LLMController(chains=self.chains, llm=llm, top_k=3)
        with self.assertRaises(ControllerException):
            _ = controller.execute(self.context)

    def test_parallelism(self):
        long = Chain(
            "long",
            "do something",
            [
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=0, message="from long chain"),
            ],
        )
        short = Chain(
            "short",
            "do something faster",
            [
                MockSkill.build_wait_skill(duration=1),
                MockSkill.build_wait_skill(duration=0, message="from short chain"),
            ],
        )
        shorter = Chain(
            "shorter", "do something even faster", [MockSkill.build_wait_skill(duration=0, message="faster")]
        )

        llm = MockLLM.from_multi_line_response(
            [
                "name: long<->score: 10<->instructions: None<->justification: because",
                "name: short<->score: 10<->instructions: None<->justification: because",
                "name: shorter<->score: 10<->instructions: None<->justification: because",
            ]
        )
        controller = LLMController(chains=[long, short, shorter], llm=llm, parallelism=True)
        context = AgentContext.from_user_message("test //", Budget(3))

        agent = Agent(controller, BasicEvaluator(), BasicFilter())
        agent.execute(context)

        remaining_duration = context.budget.remaining_duration
        self.assertFalse(context.budget.is_expired())
        self.assertLessEqual(remaining_duration, 1)
        self.assertGreaterEqual(remaining_duration, 0.5)
