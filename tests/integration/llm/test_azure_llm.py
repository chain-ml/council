import unittest

import dotenv

from council.contexts import LLMContext
from council.llm import AzureLLM, LLMMessage, LLMException, LLMCallTimeoutException
from council.utils import ParameterValueException, OsEnviron


class TestLlmAzure(unittest.TestCase):
    """requires an Azure LLM model deployed"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

    def test_basic_prompt(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]

        llm_result = self.llm.post_chat_request(LLMContext.empty(), messages)
        result = llm_result.first_choice
        print(result)
        messages.append(LLMMessage.system_message(result))
        messages.append(LLMMessage.user_message("give me another example"))
        results = self.llm.post_chat_request(LLMContext.empty(), messages)
        [print(choice) for choice in results.choices]

    @unittest.skip("Azure no longer censored this prompt")
    def test_censored_prompt(self):
        messages = [
            LLMMessage.user_message(
                "What is the easiest way for me to buy self-custodied ETH using a Canadian bank account?"
            )
        ]

        with self.assertRaises(LLMException) as e:
            self.llm.post_chat_request(LLMContext.empty(), messages)
            self.assertIn("censored", str(e))

    def test_max_token(self):
        with OsEnviron("AZURE_LLM_MAX_TOKENS", "5"):
            llm = AzureLLM.from_env()
            messages = [LLMMessage.user_message("Give me an example of a currency")]
            result = llm.post_chat_request(LLMContext.empty(), messages)
            self.assertTrue(len(result.first_choice.replace(" ", "")) <= 5 * 5)

    def test_choices(self):
        with OsEnviron("AZURE_LLM_N", "3"), OsEnviron("AZURE_LLM_TEMPERATURE", "1.0"):
            llm = AzureLLM.from_env()
            messages = [LLMMessage.user_message("Give me an example of a currency")]
            result = llm.post_chat_request(LLMContext.empty(), messages)
            self.assertEquals(3, len(result.choices))
            [print("\n- Choice:" + choice) for choice in result.choices]

    def test_invalid_temperature(self):
        with OsEnviron("AZURE_LLM_TEMPERATURE", "3.5"):
            with self.assertRaises(ParameterValueException) as cm:
                _ = AzureLLM.from_env()
            print(cm.exception)

    def test_time_out(self):
        with OsEnviron("AZURE_LLM_TIMEOUT", "1"):
            llm = AzureLLM.from_env()
            messages = [LLMMessage.user_message("Give a full explanation of quantum intrication ")]
            with self.assertRaises(LLMCallTimeoutException):
                _ = llm.post_chat_request(LLMContext.empty(), messages)
