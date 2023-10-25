import unittest

from council.controllers.llm_controller import Specialist
from council.llm import LLMAnswer


class TestLLMFallBack(unittest.TestCase):
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
