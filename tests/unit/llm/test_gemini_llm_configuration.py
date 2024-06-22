import unittest
from council.llm import GeminiLLMConfiguration
from council.utils import OsEnviron, ParameterValueException


class TestGeminiLLMConfiguration(unittest.TestCase):
    def test_model_override(self):
        with OsEnviron("GEMINI_API_KEY", "some-key"), OsEnviron("GEMINI_LLM_MODEL", "gemini-something"):
            config = GeminiLLMConfiguration.from_env()
            self.assertEqual("some-key", config.api_key.value)
            self.assertEqual("gemini-something", config.model.value)

    def test_default(self):
        config = GeminiLLMConfiguration(model="gemini-something", api_key="some-key")
        self.assertEqual(0.0, config.temperature.value)
        self.assertTrue(config.top_p.is_none())

    def test_invalid(self):
        with self.assertRaises(ParameterValueException):
            _ = GeminiLLMConfiguration(model="a-gemini-model", api_key="sk-key")
        with self.assertRaises(ParameterValueException):
            _ = GeminiLLMConfiguration(model="gemini-model", api_key="")
