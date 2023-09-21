from typing import Sequence
from unittest import TestCase

import dotenv

from council.chains import Chain
from council.contexts import AgentContext, Budget
from council.controllers import LLMController, ExecutionUnit
from council.llm import AzureLLM


class TestAzureLlmController(TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        self.chain_news = Chain(
            "chat with google news context",
            "answer any serious questions about Finance using News information",
            [],
            support_instructions=True,
        )
        self.chain_search = Chain(
            "chat with google search context",
            "answer any serious questions about Finance using a generic search engine",
            [],
        )
        self.chain_tex2sql = Chain("text2sql", "generate a sql query from a user prompt", [])
        self.chain_forecast = Chain(
            "forecast",
            "generate a forecast on timeseries data retrieve from a sql query",
            [],
            support_instructions=True,
        )

        self.chains = [
            self.chain_search,
            self.chain_news,
            self.chain_tex2sql,
            self.chain_forecast,
        ]
        dotenv.load_dotenv()

    def test_controller_chain_search(self):
        self._test_prompt("what is inflation?", [self.chain_search])
        self._test_prompt("what is APY?", [self.chain_search])

    def test_controller_chain_news(self):
        self._test_prompt("What recently impacted the USD/CAN exchange rate?", [self.chain_news])

    def test_controller_chain_news_1_choice(self):
        self._test_prompt("What recently impacted the USD/CAN exchange rate?", [self.chain_news], 1)

    def test_controller_no_chain(self):
        self._test_prompt("tell me a joke about finance", [])

    def test_controller_chain_text2sql(self):
        self._test_prompt(
            "generate a sql query to get the value of inflation for the last week", [self.chain_tex2sql], top_k=1
        )

    def test_controller_chain_forecast(self):
        result = self._test_prompt(
            "forecast the value of inflation for the next week, using one month of data",
            [self.chain_forecast],
        )
        self.assertIsNotNone(result[0].initial_state)
        print(f"Instructions: {result[0].initial_state.message}")

    def _test_prompt(self, prompt: str, expected: Sequence[Chain], top_k: int = 1000) -> List[ExecutionUnit]:
        print("\n*******")
        print(f"Prompt: {prompt}")
        controller = LLMController(chains=self.chains, llm=AzureLLM.from_env(), top_k=top_k)
        execution_context = AgentContext.from_user_message(prompt, Budget(10))
        result = controller.execute(execution_context)

        if len(expected) == 0:
            self.assertEqual(0, len(result), "no result is expected")
        else:
            self.assertEqual(min(len(self.chains), top_k), len(result), "result length")
        self.assertEqual(expected, [item.chain for item in result[: len(expected)]])
        return result
