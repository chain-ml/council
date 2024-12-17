import unittest
import dotenv

import json

from council import LLMContext
from council.llm import LLMMessage, GroqLLM
from council.utils import OsEnviron


class TestGroqLLM(unittest.TestCase):
    @staticmethod
    def get_gemma():
        dotenv.load_dotenv()
        with OsEnviron("GROQ_LLM_MODEL", "gemma2-9b-it"):
            return GroqLLM.from_env()

    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        llm = self.get_gemma()

        result = llm.post_chat_request(LLMContext.empty(), messages)
        assert "Paris" in result.choices[0]

        messages.append(LLMMessage.user_message("give a famous monument of that place"))
        result = llm.post_chat_request(LLMContext.empty(), messages)

        assert "Eiffel" in result.choices[0]

    def test_consumptions(self):
        messages = [LLMMessage.user_message("Hello how are you?")]
        llm = self.get_gemma()
        result = llm.post_chat_request(LLMContext.empty(), messages)

        assert len(result.consumptions) == 12  # call, duration, 3 token kinds, 3 cost kinds and 4 groq duration
        for consumption in result.consumptions:
            assert consumption.kind.startswith("gemma2-9b-it")

    def test_max_tokens_param(self):
        llm = self.get_gemma()
        llm.configuration.temperature.set(0.8)
        llm.configuration.max_tokens.set(7)

        messages = [LLMMessage.user_message("Hey how are you?")]
        result = llm.post_chat_request(LLMContext.empty(), messages)
        print(f"Predicted: {result.first_choice}")

    def test_json_mode(self):
        messages = [LLMMessage.user_message("Output a JSON object with the data about RPG character.")]
        llm = self.get_gemma()
        result = llm.post_chat_request(LLMContext.empty(), messages, response_format={"type": "json_object"})

        data = json.loads(result.first_choice)
        assert isinstance(data, dict)
