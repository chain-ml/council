import unittest

import dotenv

from council.llm import OpenAILLM, LLMMessage, LLMTokenLimitException


class TestLlmOpenAI(unittest.TestCase):
    """requires an OpenAI access key"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = OpenAILLM.from_env()

    def test_basic_prompt(self):
        messages = [LLMMessage.user_message("what are the largest cities in South America")]

        result = self.llm.post_chat_request(messages, model="gpt-3.5-turbo")[0]
        print(result)
        messages.append(LLMMessage.system_message(result))

        messages = [LLMMessage.user_message("what are the largest cities in South America")]
        result = self.llm.post_chat_request(messages, model="gpt-4")[0]
        print(result)
        messages.append(LLMMessage.system_message(result))
        messages.append(LLMMessage.user_message("give me another example"))
        result = self.llm.post_chat_request(messages, model="gpt-4")
        print(result)

    def test_prompt_exceed_token_limit(self):
        messages = [LLMMessage.user_message("what are the largest cities in South America")] * 500
        with self.assertRaises(LLMTokenLimitException) as cm:
            _ = self.llm.post_chat_request(messages)
        print(str(cm.exception))
