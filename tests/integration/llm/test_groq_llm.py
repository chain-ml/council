import unittest
import dotenv

from council import LLMContext
from council.llm import LLMMessage, GroqLLM
from council.utils import OsEnviron


class TestGroqLLM(unittest.TestCase):
    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("GROQ_LLM_MODEL", "llama-3.2-1b-preview"):
            instance = GroqLLM.from_env()

        result = instance.post_chat_request(LLMContext.empty(), messages)
        assert "Paris" in result.choices[0]

        messages.append(LLMMessage.user_message("give a famous monument of that place"))
        result = instance.post_chat_request(LLMContext.empty(), messages)

        assert "Eiffel" in result.choices[0]

    def test_consumptions(self):
        messages = [LLMMessage.user_message("Hello how are you?")]
        dotenv.load_dotenv()
        with OsEnviron("GROQ_LLM_MODEL", "llama-3.2-1b-preview"):
            instance = GroqLLM.from_env()
            result = instance.post_chat_request(LLMContext.empty(), messages)

            assert len(result.consumptions) == 12  # call, duration, 3 token kinds, 3 cost kinds and 4 groq duration
            for consumption in result.consumptions:
                assert consumption.kind.startswith("llama-3.2-1b-preview")
