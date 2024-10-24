import unittest

import dotenv

from council.llm import LLMMessage, AnthropicLLMConfiguration, GeminiLLMConfiguration, OpenAIChatGPTConfiguration
from council.llm.llm_message import LLMMessageData
from council.llm.llm_middleware import LLMCachingMiddleware, LLMRequest
from council.utils import OsEnviron


class TestLlmCachingMiddleware(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            self.anthropic_config = AnthropicLLMConfiguration.from_env()

        with OsEnviron("GEMINI_LLM_MODEL", "gemini-1.5-flash"):
            self.gemini_config = GeminiLLMConfiguration.from_env()

        with OsEnviron("OPENAI_LLM_MODEL", "gpt-4o-mini"):
            self.openai_config = OpenAIChatGPTConfiguration.from_env()

    def get_hash(self, messages, configuration=None):
        request = LLMRequest.default(messages)
        if configuration is not None:
            return LLMCachingMiddleware.get_hash(request, configuration)
        return LLMCachingMiddleware.get_hash(request, self.anthropic_config)

    def test_message_hashing(self):
        messages_v1 = [LLMMessage.system_message("System message"), LLMMessage.user_message("User message")]
        messages_v1_1 = [LLMMessage.system_message("System message"), LLMMessage.user_message("UsEr MesSagE   ")]
        messages_v2 = [LLMMessage.system_message("System message"), LLMMessage.user_message("Different user message")]
        messages_v3 = [
            LLMMessage.system_message("System message"),
            LLMMessage.user_message("User message", data=[LLMMessageData(content="data", mime_type="")]),
        ]
        messages_v3_1 = [
            LLMMessage.system_message("System message"),
            LLMMessage.user_message("User message   ", data=[LLMMessageData(content="data   ", mime_type="")]),
        ]
        messages_v4 = [
            LLMMessage.system_message("System message"),
            LLMMessage.user_message("User message", data=[LLMMessageData(content="different data", mime_type="")]),
        ]

        self.assertEqual(self.get_hash(messages_v1), self.get_hash(messages_v1))
        self.assertEqual(self.get_hash(messages_v1), self.get_hash(messages_v1_1))
        self.assertNotEqual(self.get_hash(messages_v1), self.get_hash(messages_v2))
        self.assertNotEqual(self.get_hash(messages_v2), self.get_hash(messages_v3))
        self.assertEqual(self.get_hash(messages_v3), self.get_hash(messages_v3_1))
        self.assertNotEqual(self.get_hash(messages_v3_1), self.get_hash(messages_v4))

    def test_config_hashing(self):
        messages = [LLMMessage.system_message("System message"), LLMMessage.user_message("User message")]

        self.assertEqual(self.get_hash(messages, self.anthropic_config), self.get_hash(messages, self.anthropic_config))
        self.assertEqual(self.get_hash(messages, self.gemini_config), self.get_hash(messages, self.gemini_config))
        self.assertEqual(self.get_hash(messages, self.openai_config), self.get_hash(messages, self.openai_config))

        self.assertNotEqual(self.get_hash(messages, self.anthropic_config), self.get_hash(messages, self.gemini_config))
        self.assertNotEqual(self.get_hash(messages, self.anthropic_config), self.get_hash(messages, self.openai_config))
        self.assertNotEqual(self.get_hash(messages, self.gemini_config), self.get_hash(messages, self.openai_config))
