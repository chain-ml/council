import unittest

import dotenv

from council import LLMContext
from council.llm import LLMMessage, OllamaLLM
from council.utils import OsEnviron


class TestOllamaLLM(unittest.TestCase):
    def setUp(self):
        self.llama_32 = self.get_llama_32()

    @staticmethod
    def get_llama_32():
        dotenv.load_dotenv()
        with OsEnviron("OLLAMA_LLM_MODEL", "llama3.2"):
            return OllamaLLM.from_env()

    def test_single_message(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        result_v1 = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert "Paris" in result_v1.first_choice

        result_v2 = self.llama_32.post_chat_request(LLMContext.empty(), messages)

        assert result_v1.first_choice == result_v2.first_choice  # same seed

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

    def test_pull(self):
        response = self.llama_32.client.delete(self.llama_32.model_name)
        assert response["status"] == "success"

        response = self.llama_32.pull()
        assert response["status"] == "success"

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

    def test_num_predict_param(self):
        llm = self.get_llama_32()
        llm.configuration.num_predict.set(5)

        messages = [LLMMessage.user_message("Hey how are you?")]
        result = llm.post_chat_request(LLMContext.empty(), messages)
        print(f"Predicted: {result.first_choice}")

    def test_stop_param(self):
        ending = "and they lived happily ever after"
        llm = self.get_llama_32()
        llm.configuration.temperature.set(0.5)
        llm.configuration.stop.set(ending)

        messages = [LLMMessage.user_message(f"Write a 50 words story ending with `{ending}`")]
        result = llm.post_chat_request(LLMContext.empty(), messages)
        print(f"Predicted: {result.first_choice}")
        print(f"For ending: {ending}")
