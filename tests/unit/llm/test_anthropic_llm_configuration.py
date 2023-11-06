import unittest
from council.llm import AnthropicLLMConfiguration
from council.utils import OsEnviron, ParameterValueException


class TestAnthropicLLMConfiguration(unittest.TestCase):
    def test_model_override(self):
        with OsEnviron("ANTHROPIC_API_KEY", "sk-key"), OsEnviron("ANTHROPIC_LLM_MODEL", "claude-not-default"):
            config = AnthropicLLMConfiguration.from_env()
            self.assertEqual("sk-key", config.api_key.value)
            self.assertEqual("claude-not-default", config.model.value)

    def test_default(self):
        config = AnthropicLLMConfiguration(model="claude-model", max_tokens=300, api_key="sk-key")
        self.assertEqual(0.0, config.temperature.value)
        self.assertEqual(300, config.max_tokens.value)
        self.assertTrue(config.top_p.is_none())

    def test_invalid(self):
        with self.assertRaises(ParameterValueException):
            _ = AnthropicLLMConfiguration(model="a-claude-model", api_key="sk-key", max_tokens=300)
        with self.assertRaises(ParameterValueException):
            _ = AnthropicLLMConfiguration(model="claude-model", api_key="a-sk-key", max_tokens=300)
        with self.assertRaises(ParameterValueException):
            _ = AnthropicLLMConfiguration(model="claude-model", api_key="sk-key", max_tokens=0)
