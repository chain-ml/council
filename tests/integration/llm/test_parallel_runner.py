import unittest

import dotenv
from typing import Sequence

from pydantic import Field

from council.llm import (
    OpenAILLM,
    LLMFunction,
    JSONResponseParser,
)
from council.llm.llm_function.parallel_runner import ParallelRunner
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


def response_reduce(results: Sequence[Response]) -> Response:
    aggregated_reasoning = ""
    running_average_random_number = 0.0

    for i, result in enumerate(results, start=1):
        aggregated_reasoning += result.reasoning
        running_average_random_number += (result.random_number - running_average_random_number) / i

    return Response(reasoning=aggregated_reasoning, random_number=int(running_average_random_number))


class TestParallelRunner(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("OPENAI_LLM_MODEL", "gpt-4o-mini"), OsEnviron("OPENAI_LLM_TEMPERATURE", "1.5"):
            self.llm = OpenAILLM.from_env()

    def test_parallel_runner(self):
        llm_func = LLMFunction(
            self.llm,
            Response.from_response,
            system_message=SYSTEM_PROMPT.format(response_template=Response.to_response_template()),
            # TODO: Template.format_for(SYSTEM_PROMPT)?
        )

        runner = ParallelRunner(llm_func.execute, n=3, reduce=response_reduce)
        result = runner.execute(response_format={"type": "json_object"})
        print(result)

        for result in runner.last_results:
            print(result)
