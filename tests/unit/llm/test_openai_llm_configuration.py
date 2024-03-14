import unittest
from council.llm import OpenAILLMConfiguration
from council.utils import OsEnviron, ParameterValueException


class TestOpenAILLMConfiguration(unittest.TestCase):
    def test_model_default_value(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            config = OpenAILLMConfiguration.from_env()
            self.assertEqual("sk-key", config.api_key.value)
            self.assertEqual("gpt-3.5-turbo", config.model.value)

    def test_model_override(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"), OsEnviron("OPENAI_LLM_MODEL", "gpt-not-default"):
            config = OpenAILLMConfiguration.from_env()
            self.assertEqual("sk-key", config.api_key.value)
            self.assertEqual("gpt-not-default", config.model.value)

    def test_default(self):
        config = OpenAILLMConfiguration(model="gpt-model", api_key="sk-key", api_host="https://api.openai.com")
        self.assertEqual(0.0, config.temperature.value)
        self.assertEqual(1, config.n.value)
        self.assertTrue(config.top_p.is_none())
        self.assertTrue(config.frequency_penalty.is_none())
        self.assertTrue(config.presence_penalty.is_none())

    def test_invalid(self):
        with self.assertRaises(ParameterValueException):
            _ = OpenAILLMConfiguration(model="a-gpt-model", api_key="sk-key", api_host="https://api.openai.com")
        with self.assertRaises(ParameterValueException):
            _ = OpenAILLMConfiguration(model="gpt-model", api_key="a-sk-key", api_host="https://api.openai.com")
        with self.assertRaises(ParameterValueException):
            _ = OpenAILLMConfiguration(model="gpt-model", api_key="sk-key", api_host="api.openai.com")
