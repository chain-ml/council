import unittest

from council.agents import Agent
from council.chains import Chain
from council.controllers import LLMController
from council.contexts import AgentContext, Budget
from council.evaluators import BasicEvaluator
from council.filters import BasicFilter
from council.mocks import MockLLM, MockSkill, MockMultipleResponses
from council.controllers.llm_controller import Specialist
from council.llm import LLMAnswer


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

    def test_llm_answer(self):
        llma = LLMAnswer(Specialist)
        print("\n")
        print("Automatic Prompt for line parsing:\n")
        print(llma.to_prompt())
        print("Automatic Prompt for yaml parsing:\n")
        print(llma.to_yaml_prompt())

    def test_llm_parse_line_answer(self):
        llma = LLMAnswer(Specialist)
        print("\n")
        print(llma.parse_line("Name: first<->Score: 10<->Instructions: None<->Justification: because"))
        print(llma.parse_line("Instructions: None<->Name: first<->Score: ABC<->Justification: because"))

        cs = llma.to_object("Instructions: None<->nAme: first<->Score: 10<->Justification: because")
        self.assertEqual(cs.score, 10)

    def test_llm_parse_yaml_answer(self):
        llma = LLMAnswer(Specialist)

        bloc = """
    ```yaml
    ControllerScore:
      name: first
      score: 10
      instructions: do this
      justification: because
    ```"""
        print(llma.parse_yaml_bloc(bloc))

    def test_plan_parse_no_matching_chain(self):
        llm_responses = [
            [
                "name: first<->score: 10<->instructions: None<->justification: because",
                "name: secondDoesNotExists<->score: 6<->instructions: None<->justification: because",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
            [
                "name: first<->score: 4<->instructions: None<->justification: because",
                "name: second<->score: 6<->instructions: None<->justification: because",
                "name: third<->score: 2<->instructions: None<->justification: because",
            ],
        ]

        llm = MockLLM(action=MockMultipleResponses(responses=llm_responses).call)

        controller = LLMController(chains=self.chains, llm=llm, top_k=3)
        result = controller.execute(self.context)
        self.assertEqual(["second", "first", "third"], [item.chain.name for item in result])

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
