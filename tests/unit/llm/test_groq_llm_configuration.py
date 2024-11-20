import unittest
from council.llm import GroqLLMConfiguration
from council.utils import OsEnviron, ParameterValueException


class TestGroqLLMConfiguration(unittest.TestCase):
    def test_model_override(self):
        with OsEnviron("GROQ_API_KEY", "some-key"), OsEnviron("GROQ_LLM_MODEL", "llama-something"):
            config = GroqLLMConfiguration.from_env()
            self.assertEqual("some-key", config.api_key.value)
            self.assertEqual("llama-something", config.model.value)

    def test_default(self):
        config = GroqLLMConfiguration(model="llama-something", api_key="some-key")
        self.assertEqual(0.0, config.temperature.value)
        self.assertTrue(config.top_p.is_none())

    def test_invalid(self):
        with self.assertRaises(ParameterValueException):
            _ = GroqLLMConfiguration(model="llama-something", api_key="")
