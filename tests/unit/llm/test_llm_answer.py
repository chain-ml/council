import unittest
from typing import Iterable, List

from council.controllers.llm_controller import Specialist
from council.llm import LLMAnswer, LLMParsingException


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

        cs: Specialist = llma.to_object("Instructions: None<->nAme: first<->Score: 10<->Justification: because")
        self.assertEqual(10, cs.score)

        with self.assertRaises(LLMParsingException) as e:
            _ = llma.to_object("Instructions: None<->nAme: first<->Score: 20<->Justification: because")
        print(f"exception: {e.exception}")

    def test_llm_parse_yaml_answer(self):
        llma = LLMAnswer(Specialist)

        bloc = """
```yaml
name: first
score: 10
instructions: do this
justification: because
```
"""
        result = llma.parse_yaml_bloc(bloc)
        instance = Specialist(**result)
        self.assertEqual(instance.name, "first")

    def test_parse_dict(self):
        instance = LLMAnswer(Specialist)
        bloc = """
- name: first
  score: 10
  instructions:
  - do
  - this 
  justification: because
"""
        print(instance.parse_yaml_list(bloc))
