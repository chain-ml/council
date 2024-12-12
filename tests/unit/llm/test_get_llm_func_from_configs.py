import os
import shutil
import unittest

from council.llm import LLMCacheControlData
from council.llm.llm_function import LLMFunctionWithPrompt
from council.llm.llm_function import StringResponseParser
from council.utils import OsEnviron
from tests import get_data_filename
from tests.unit import LLMPrompts, LLMModels


class TestLLMFunctionWithPromptFromConfigs(unittest.TestCase):
    def setUp(self):
        # Create temporary directories
        module_path = os.path.dirname(__file__)

        self.data_dir = os.path.join(module_path, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.another_data_dir = os.path.join(module_path, "another_data", "inner")
        os.makedirs(self.another_data_dir, exist_ok=True)

        llm_config_path = get_data_filename(LLMModels.OpenAI)
        prompt_path = get_data_filename(LLMPrompts.sample)

        shutil.copy(prompt_path, os.path.join(self.data_dir, "llm-prompt.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir, "llm-config.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir, "llm-config-v2.yaml"))

        shutil.copy(prompt_path, os.path.join(self.another_data_dir, "llm-prompt.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.another_data_dir, "llm-config.yaml"))
        shutil.copy(prompt_path, os.path.join(self.another_data_dir, "llm-prompt-v2.yaml"))

    def tearDown(self):
        shutil.rmtree(self.data_dir)
        shutil.rmtree(self.another_data_dir)

    def test_default(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(response_parser=StringResponseParser.from_response)

        assert isinstance(llm_function, LLMFunctionWithPrompt)

    def test_override_llm(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(
                response_parser=StringResponseParser.from_response, llm_path="llm-config-v2.yaml"
            )

        assert isinstance(llm_function, LLMFunctionWithPrompt)

    def test_override_base_path(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(
                response_parser=StringResponseParser.from_response,
                path_prefix=self.another_data_dir,
            )

        assert isinstance(llm_function, LLMFunctionWithPrompt)

    def test_override_base_path_and_prompt(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(
                response_parser=StringResponseParser.from_response,
                path_prefix=self.another_data_dir,
                prompt_config_path="llm-prompt-v2.yaml",
            )
        assert isinstance(llm_function, LLMFunctionWithPrompt)

    def test_with_params(self):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(response_parser=StringResponseParser.from_response)

        assert isinstance(llm_function, LLMFunctionWithPrompt)
        assert llm_function._max_retries == 3

        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            llm_function = LLMFunctionWithPrompt.from_configs(
                response_parser=StringResponseParser.from_response,
                max_retries=42,
                system_prompt_caching=True,
            )

        assert isinstance(llm_function, LLMFunctionWithPrompt)
        assert llm_function._max_retries == 42
        assert len(llm_function._messages[0].data) == 1
        assert isinstance(llm_function._messages[0].data[0], LLMCacheControlData)
