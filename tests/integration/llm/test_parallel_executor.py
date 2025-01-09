import unittest
from dataclasses import dataclass

import dotenv
from typing import Sequence

from pydantic import Field

from council.llm import OpenAILLM, LLMFunction, JSONResponseParser, ParallelExecutor
from council.utils import OsEnviron

SYSTEM_PROMPT = """
Generate an random number.

# Response template
{response_template}
"""


class Response(JSONResponseParser):
    reasoning: str = Field(..., description="The reasoning behind the response")
    random_number: int = Field(..., description="A random number between 1 and 100")

    def __str__(self) -> str:
        return f"Reasoning: {self.reasoning}\nRandom number: {self.random_number}"


@dataclass
class AggregatedResponse:
    random_number: int


def reduce_to_response(results: Sequence[Response]) -> Response:
    aggregated_reasoning = ""
    running_average_random_number = 0.0

    for i, result in enumerate(results, start=1):
        aggregated_reasoning += " " + result.reasoning
        running_average_random_number += (result.random_number - running_average_random_number) / i

    return Response(reasoning=aggregated_reasoning, random_number=int(running_average_random_number))


def reduce_to_aggregated_response(results: Sequence[Response]) -> AggregatedResponse:
    average = sum(result.random_number for result in results) / len(results)

    return AggregatedResponse(int(average))


class TestParallelExecutor(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("OPENAI_LLM_MODEL", "gpt-4o-mini"), OsEnviron("OPENAI_LLM_TEMPERATURE", "1.5"):
            self.llm = OpenAILLM.from_env()

    def get_llm_func(self) -> LLMFunction[Response]:
        return LLMFunction(self.llm, Response.from_response, system_message=Response.format(SYSTEM_PROMPT))

    def test_with_n(self):
        llm_func = self.get_llm_func()

        executor = ParallelExecutor(llm_func.execute, reduce=reduce_to_response, n=3)
        results = executor.execute(response_format={"type": "json_object"})

        self.assertEquals(len(results), 3)

        for result in results:
            self.assertIsInstance(result, Response)

        result = executor.reduce(results)
        self.assertIsInstance(result, Response)

    def test_with_executes(self):
        llm_func_v1, llm_func_v2 = self.get_llm_func(), self.get_llm_func()

        executor = ParallelExecutor([llm_func_v1.execute, llm_func_v2.execute], reduce=reduce_to_aggregated_response)
        results = executor.execute(response_format={"type": "json_object"})

        self.assertEquals(len(results), 2)

        for result in results:
            self.assertIsInstance(result, Response)

        result = executor.reduce(results)
        self.assertIsInstance(result, AggregatedResponse)
