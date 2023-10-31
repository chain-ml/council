import unittest

import dotenv
from council import LLMContext
from council.llm import LLMMessage, AnthropicLLM


class TestAnthropicLLM(unittest.TestCase):
    def test_completion(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        instance = AnthropicLLM.from_env()
        context = LLMContext.empty()
        result = instance.post_chat_request(context, messages)

        assert "Paris" in result.choices[0]
