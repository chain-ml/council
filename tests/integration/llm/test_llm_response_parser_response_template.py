import unittest

import dotenv
from typing import Type

from pydantic import Field

from council import OpenAILLM
from council.llm_function.llm_function import LLMFunction, LLMFunctionResponse
from council.llm_function.llm_response_parser import (
    YAMLBlockResponseParser,
    YAMLResponseParser,
    JSONBlockResponseParser,
    JSONResponseParser,
)
from council.utils import OsEnviron

SYSTEM_PROMPT = """
Generate an RPG character.

# Response template
{response_template}
"""


class Character:
    name: str = Field(..., description="The name of the character")
    power: float = Field(..., ge=0, le=1, description="The power of the character, float from 0 to 1")
    role: str = Field(..., description="The role of the character")

    def __str__(self):
        return f"Name: {self.name}, Power: {self.power}, Role: {self.role}"


class YAMLBlockCharacter(YAMLBlockResponseParser, Character):
    pass


class YAMLCharacter(YAMLResponseParser, Character):
    pass


class JSONBlockCharacter(JSONBlockResponseParser, Character):
    pass


class JSONCharacter(JSONResponseParser, Character):
    pass


class TestLLMResponseParserResponseTemplate(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("OPENAI_LLM_MODEL", "gpt-4o-mini"):
            self.llm = OpenAILLM.from_env()

    def _check_response(self, llm_function_response: LLMFunctionResponse, expected_response_type: Type):
        response = llm_function_response.response
        self.assertIsInstance(response, expected_response_type)
        print("", response, sep="\n")

        self.assertTrue(len(llm_function_response.consumptions) == 8)  # no self-correction

    def test_yaml_block_response_parser(self):
        llm_func = LLMFunction(
            self.llm,
            YAMLBlockCharacter.from_response,
            system_message=SYSTEM_PROMPT.format(response_template=YAMLBlockCharacter.to_response_template()),
        )
        llm_function_response = llm_func.execute_with_llm_response(user_message="Create wise old wizard")

        self._check_response(llm_function_response, YAMLBlockCharacter)

    def test_yaml_response_parser(self):
        llm_func = LLMFunction(
            self.llm,
            YAMLCharacter.from_response,
            system_message=SYSTEM_PROMPT.format(response_template=YAMLCharacter.to_response_template()),
        )
        llm_function_response = llm_func.execute_with_llm_response(user_message="Create strong warrior")

        self._check_response(llm_function_response, YAMLCharacter)

    def test_json_block_response_parser(self):
        llm_func = LLMFunction(
            self.llm,
            JSONBlockCharacter.from_response,
            system_message=SYSTEM_PROMPT.format(response_template=JSONBlockCharacter.to_response_template()),
        )
        llm_function_response = llm_func.execute_with_llm_response(user_message="Create kind dwarf")

        self._check_response(llm_function_response, JSONBlockCharacter)

    def test_json_response_parser(self):
        llm_func = LLMFunction(
            self.llm,
            JSONCharacter.from_response,
            system_message=SYSTEM_PROMPT.format(response_template=JSONCharacter.to_response_template()),
        )
        llm_function_response = llm_func.execute_with_llm_response(
            user_message="Create shadow thief", response_format={"type": "json_object"}
        )

        self._check_response(llm_function_response, JSONCharacter)
