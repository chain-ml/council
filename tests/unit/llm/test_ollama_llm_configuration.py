import unittest
from council.llm.ollama_llm_configuration import OllamaLLMConfiguration
from council.utils import OsEnviron


class TestOllamaLLMConfiguration(unittest.TestCase):
    def test_model_override(self):
        with OsEnviron("OLLAMA_LLM_MODEL", "llama3.2"):
            config = OllamaLLMConfiguration.from_env()
            self.assertEqual("llama3.2", config.model.value)

    def test_keep_alive_override(self):
        with OsEnviron("OLLAMA_LLM_MODEL", "llama3.2"), OsEnviron("OLLAMA_KEEP_ALIVE", "10m"):
            config = OllamaLLMConfiguration.from_env()
            self.assertEqual("llama3.2", config.model.value)

            self.assertEqual("10m", config.keep_alive)

    def test_default(self):
        config = OllamaLLMConfiguration(model="llama3.2")
        self.assertEqual(0.0, config.temperature.value)
        self.assertIsNone(config.keep_alive)
        self.assertTrue(config.top_p.is_none())
