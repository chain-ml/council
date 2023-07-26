from typing import List
from unittest import TestCase

import dotenv

from council.chains import Chain
from council.contexts import ChatHistory, AgentContext
from council.controllers import LLMController
from council.llm import AzureLLM
from council.runners import Budget


class TestAzureLlmController(TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        self.chain_news = Chain(
            "chat with google news context",
            "answer a question about Finance using News information",
            [],
        )
        self.chain_search = Chain(
            "chat with google search context",
            "answer a question about Finance using a generic search engine",
            [],
        )
        self.chain_tex2sql = Chain("text2sql", "generate a sql query from a user prompt", [])
        self.chain_forecast = Chain(
            "forecast",
            "generate a forecast on timeseries data retrieve from a sql query",
            [],
        )

        self.chains = [
            self.chain_search,
            self.chain_news,
            self.chain_tex2sql,
            self.chain_forecast,
        ]
        dotenv.load_dotenv()

    def test_controller(self):
        self._test_prompt("what is inflation?", [self.chain_search])
        self._test_prompt("What recently impacted the USD/CAN exchange rate?", [self.chain_news])
        self._test_prompt("tell me a joke about finance", [])
        self._test_prompt("what is APY?", [self.chain_search])
        self._test_prompt(
            "generate a sql query to get the value of inflation for the last week",
            [self.chain_tex2sql],
        )
        self._test_prompt(
            "forecast the value of inflation for the next week, using one month of data",
            [self.chain_forecast],
        )

    def _test_prompt(self, prompt: str, expected: List[Chain]):
        print("*******")
        print(prompt)
        controller = LLMController(AzureLLM.from_env())
        chat_history = ChatHistory.from_user_message(prompt)
        execution_context = AgentContext(chat_history)
        result = controller.get_plan(execution_context, self.chains, Budget(10))
        self.assertLessEqual(len(expected), len(result), "result length")
        self.assertEqual(expected, [item.chain for item in result[: len(expected)]])
