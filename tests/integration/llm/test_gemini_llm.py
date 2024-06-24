import unittest

import dotenv
from council import LLMContext
from council.llm import LLMMessage, GeminiLLM
from council.utils import OsEnviron


class TestAnthropicLLM(unittest.TestCase):
    def test_completion(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-flash"):
            instance = GeminiLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.choices[0]

    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.0-pro"):
            instance = GeminiLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.choices[0]

        messages.append(LLMMessage.user_message("give a famous monument of that place"))
        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-pro"):
            instance = GeminiLLM.from_env()
            result = instance.post_chat_request(context, messages)

            assert "Eiffel" in result.choices[0]
