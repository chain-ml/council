import unittest

import dotenv

from council.llm import AzureLLM, AzureConfiguration, LLMMessage, LLMException


class TestLlmAzure(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        config = AzureConfiguration.from_env()
        self.llm = AzureLLM(config)

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
