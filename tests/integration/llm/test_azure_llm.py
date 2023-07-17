import os
import unittest

import dotenv

from council.llm import AzureLLM, LLMMessage, LLMException


class TestLlmAzure(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

    def test_basic_prompt(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]

        result = self.llm.post_chat_request(messages)
        print(result)
        messages.append(LLMMessage.system_message(result))
        messages.append(LLMMessage.user_message("give me another example"))
        result = self.llm.post_chat_request(messages)
        print(result)

    def test_censored_prompt(self):
        messages = [
            LLMMessage.user_message(
                "What is the easiest way for me to buy self-custodied ETH using a Canadian bank account?"
            )
        ]

        with self.assertRaises(LLMException) as e:
            self.llm.post_chat_request(messages)
            self.assertIn("censored", str(e))

    def test_max_token(self):
        os.environ["AZURE_LLM_MAX_TOKENS"] = "5"

        try:
            llm = AzureLLM.from_env()
            messages = [LLMMessage.user_message("Give me an example of a currency")]
            result = llm.post_chat_request(messages)
            self.assertTrue(len(result.replace(" ", "")) < 5 * 5)
        finally:
            del os.environ["AZURE_LLM_MAX_TOKEN"]

    def test_choices(self):
        os.environ["AZURE_LLM_N"] = "3"
        os.environ["AZURE_LLM_TEMPERATURE"] = "1.0"

        try:
            llm = AzureLLM.from_env()
            messages = [LLMMessage.user_message("Give me an example of a currency")]
            result = llm.post_chat_request(messages)
            [print("\n- Choice:" + choice) for choice in result.split("--- choices ---")]
        finally:
            del os.environ["AZURE_LLM_N"]
            del os.environ["AZURE_LLM_TEMPERATURE"]
