import time
import unittest
import dotenv

from council import LLMContext
from council.llm import LLMMessage, AnthropicLLM
from council.llm.llm_message import LLMCacheControlData
from council.utils import OsEnviron

from tests import get_data_filename


class TestAnthropicLLM(unittest.TestCase):
    large_content = "Paris is capital of France" * 300  # caching from 2k tokens for haiku

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

    def test_no_cache_system_prompt(self):
        dotenv.load_dotenv()
        messages = [
            LLMMessage.system_message(self.large_content),
            LLMMessage.user_message("What's the capital of France?"),
        ]
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert all(not key.startswith("cache_") for key in result.raw_response["usage"])

    def test_cache_system_prompt(self):
        dotenv.load_dotenv()
        messages = [
            LLMMessage.system_message(self.large_content, data=[LLMCacheControlData.ephemeral()]),
            LLMMessage.user_message("What's the capital of France?"),
        ]
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert result.raw_response["usage"]["cache_creation_input_tokens"] > 0

            time.sleep(1)
            result = instance.post_chat_request(context, messages)
            assert result.raw_response["usage"]["cache_read_input_tokens"] > 0

    def test_cache_user_prompt(self):
        dotenv.load_dotenv()
        messages = [
            LLMMessage.system_message(self.large_content),
            LLMMessage.user_message("What's the capital of France?", data=[LLMCacheControlData.ephemeral()]),
        ]
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            instance = AnthropicLLM.from_env()
            context = LLMContext.empty()
            result = instance.post_chat_request(context, messages)

            assert result.raw_response["usage"]["cache_creation_input_tokens"] > 0

            time.sleep(1)
            result = instance.post_chat_request(context, messages)
            assert result.raw_response["usage"]["cache_read_input_tokens"] > 0
