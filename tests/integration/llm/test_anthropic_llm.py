import unittest
import dotenv

from council import LLMContext
from council.llm import LLMMessage, AnthropicLLM
from council.utils import OsEnviron

from tests import get_data_filename


class TestAnthropicLLM(unittest.TestCase):
    def test_completion(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-2"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

    def test_message(self):
        messages = [LLMMessage.user_message("what is the capital of France?")]
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-2.1"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20240620"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert "Paris" in result.first_choice

    def test_with_png_image(self):
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20240620"):
            instance = AnthropicLLM.from_env()
            message = LLMMessage.user_message("What is in the image?")
            message.add_content(path=get_data_filename("Gfp-wisconsin-madison-the-nature-boardwalk.png"))
            result = instance.post_chat_request(LLMContext.empty(), [message])
            print(result.first_choice)
