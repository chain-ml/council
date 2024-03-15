import unittest

import dotenv
from council import LLMContext
from council.llm import LLMMessage, AnthropicLLM
from council.utils import OsEnviron


class TestAnthropicLLM(unittest.TestCase):
    def test_completion(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-2"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.choices[0]

    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-2.1"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.choices[0]

        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.choices[0]
