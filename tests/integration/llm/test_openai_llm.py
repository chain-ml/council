import unittest

import dotenv

from council.llm import OpenAILLM, OpenAIConfiguration, LLMMessage


class TestLlmOpenAI(unittest.TestCase):
    """requires an OpenAI access key"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        config = OpenAIConfiguration.from_env()
        self.llm = OpenAILLM(config)

    def test_basic_prompt(self):
        messages = [LLMMessage.user_message("what are the largest cities in South America")]

        result = self.llm.post_chat_request(messages, model="gpt-3.5-turbo")
        print(result)
        messages.append(LLMMessage.system_message(result))

        messages = [LLMMessage.user_message("what are the largest cities in South America")]
        result = self.llm.post_chat_request(messages, model="gpt-4")
        print(result)
        messages.append(LLMMessage.system_message(result))
        messages.append(LLMMessage.user_message("give me another example"))
        result = self.llm.post_chat_request(messages, model="gpt-4")
        print(result)
