import unittest
from typing import Sequence

from council.chains import Chain
from council.controllers import LLMController, LLMControllerAnswer
from council.contexts import AgentContext, Budget
from council.controllers.llm_controller import Specialist
from council.llm import LLMMessage
from council.mocks import MockLLM


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
        llma = LLMControllerAnswer(Specialist)
        print("\n")
        print("Automatic Prompt for line parsing:\n")
        print(llma.to_prompt())
        print("Automatic Prompt for yaml parsing:\n")
        print(llma.to_yaml_prompt())

    def test_llm_parse_line_answer(self):
        llma = LLMControllerAnswer(Specialist)
        print("\n")
        print(llma.parse_line("Name: first<->Score: 10<->Instructions: None<->Justification: because"))
        print(llma.parse_line("Instructions: None<->Name: first<->Score: ABC<->Justification: because"))

        cs = llma.to_object("Instructions: None<->nAme: first<->Score: 10<->Justification: because")
        self.assertEqual(cs.score, 10)

    def test_llm_parse_yaml_answer(self):
        llma = LLMControllerAnswer(Specialist)

        bloc = """
    ```yaml
    ControllerScore:
      name: first
      score: 10
      instructions: do this
      justification: because
    ```"""
        print(llma.parse_yaml(llma.try_extract_bloc(bloc, "yaml")))

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

        class LLMResponses:
            def __init__(self, responses):
                self._count = 0
                self._responses = ["\n".join(resp) for resp in responses]

            def call(self, _messages: Sequence[LLMMessage]) -> Sequence[str]:
                if self._count < len(self._responses):
                    self._count += 1
                return [self._responses[self._count - 1]]

        llm = MockLLM(action=LLMResponses(responses=llm_responses).call)

        controller = LLMController(chains=self.chains, llm=llm, top_k=3)
        result = controller.execute(self.context)
        self.assertEqual(["second", "first", "third"], [item.chain.name for item in result])
