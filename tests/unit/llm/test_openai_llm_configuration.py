import unittest
from council.llm import OpenAILLMConfiguration
from council.utils import OsEnviron


class TestOpenAILLMConfiguration(unittest.TestCase):
    def test_model_default_value(self):
        with OsEnviron("OPENAI_API_KEY", "akey"):
            config = OpenAILLMConfiguration.from_env()
            self.assertEqual("akey", config.api_key.value)
            self.assertEqual("gpt-3.5-turbo", config.model.value)

    def test_model_override(self):
        with OsEnviron("OPENAI_API_KEY", "akey"), OsEnviron("OPENAI_LLM_MODEL", "amodel"):
            config = OpenAILLMConfiguration.from_env()
            self.assertEqual("akey", config.api_key.value)
            self.assertEqual("amodel", config.model.value)
