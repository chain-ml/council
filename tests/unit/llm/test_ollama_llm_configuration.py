import unittest
from council.llm.ollama_llm_configuration import OllamaLLMConfiguration
from council.utils import OsEnviron


class TestOllamaLLMConfiguration(unittest.TestCase):
    def test_model_override(self):
        with OsEnviron("OLLAMA_LLM_MODEL", "llama3.2"):
            config = OllamaLLMConfiguration.from_env()
            self.assertEqual("llama3.2", config.model.value)
