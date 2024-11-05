import unittest
import dotenv

from council import LLMContext
from council.llm import LLMMessage, OllamaLLM
from council.utils import OsEnviron


class TestOllamaLLM(unittest.TestCase):
    def setUp(self):
        dotenv.load_dotenv()
        with OsEnviron("OLLAMA_LLM_MODEL", "llama3.2"):
            self.llama_32 = OllamaLLM.from_env()

    def test_single_message(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        result = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert "Paris" in result.first_choice

    def test_messages(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        result = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert "Paris" in result.first_choice

        messages.extend(
            [
                LLMMessage.assistant_message(result.first_choice),
                LLMMessage.user_message("Give a famous monument of that place"),
            ]
        )
        result = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert "Eiffel" in result.first_choice

    def test_load(self):
        response = self.llama_32.load()

        assert response["done_reason"] == "load"

    def test_unload(self):
        response = self.llama_32.unload()

        assert response["done_reason"] == "unload"

    def test_consumptions(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        result = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert len(result.consumptions) == 9  # 2 base, 3 tokens and 4 ollama specific durations
        for consumption in result.consumptions:
            assert consumption.kind.startswith("llama3.2")
