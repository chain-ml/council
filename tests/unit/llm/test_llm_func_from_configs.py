import os
import shutil
import unittest

from council.llm import LLMCacheControlData, LLMFunctionWithPrompt, OpenAIChatGPTConfiguration
from council.utils import OsEnviron
from tests import get_data_filename
from tests.unit import LLMPrompts, LLMModels


class TestLLMFunctionWithPromptFromConfigs(unittest.TestCase):
    def setUp(self):
        # Create temporary directories
        module_path = os.path.dirname(__file__)

        self.data_dir = os.path.join(module_path, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.data_dir_internal = os.path.join(module_path, "another_data", "inner")
        os.makedirs(self.data_dir_internal, exist_ok=True)

        self.data_dir_external = os.path.join(module_path, "..", "external_data")
        os.makedirs(self.data_dir_external, exist_ok=True)

        llm_config_path = get_data_filename(LLMModels.OpenAI)
        prompt_path = get_data_filename(LLMPrompts.sample)

        shutil.copy(prompt_path, os.path.join(self.data_dir, "llm-prompt.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir, "llm-config.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir, "llm-config-v2.yaml"))

        shutil.copy(prompt_path, os.path.join(self.data_dir_internal, "llm-prompt.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir_internal, "llm-config.yaml"))
        shutil.copy(prompt_path, os.path.join(self.data_dir_internal, "llm-prompt-v2.yaml"))

        shutil.copy(prompt_path, os.path.join(self.data_dir_external, "llm-prompt.yaml"))
        shutil.copy(llm_config_path, os.path.join(self.data_dir_external, "llm-config.yaml"))

    def tearDown(self):
        shutil.rmtree(self.data_dir)
        shutil.rmtree(self.data_dir_internal)
        shutil.rmtree(self.data_dir_external)

    @classmethod
    def create_func(cls, *args, **kwargs):
        with OsEnviron("OPENAI_API_KEY", "sk-key"):
            return LLMFunctionWithPrompt.string_from_configs(*args, **kwargs)

    @classmethod
    def create_func_and_assert(cls, *args, **kwargs):
        llm_function = cls.create_func(*args, **kwargs)

        assert isinstance(llm_function, LLMFunctionWithPrompt)
        assert llm_function._max_retries == 3
        assert isinstance(llm_function._llm_config, OpenAIChatGPTConfiguration)
        assert len(llm_function._messages) == 1

    def test_default(self):
        self.create_func_and_assert(path_prefix=self.data_dir)

    def test_override_llm(self):
        self.create_func_and_assert(path_prefix=self.data_dir, llm_path="llm-config-v2.yaml")

    def test_override_base_path_internal(self):
        self.create_func_and_assert(path_prefix=self.data_dir_internal)

    def test_override_base_path_and_prompt(self):
        self.create_func_and_assert(path_prefix=self.data_dir_internal, prompt_config_path="llm-prompt-v2.yaml")

    def test_override_base_path_external(self):
        self.create_func_and_assert(path_prefix=self.data_dir_external)

    def test_with_params(self):
        llm_function = self.create_func(path_prefix=self.data_dir, max_retries=42, system_prompt_caching=True)

        assert isinstance(llm_function, LLMFunctionWithPrompt)
        assert llm_function._max_retries == 42
        assert len(llm_function._messages[0].data) == 1
        assert isinstance(llm_function._messages[0].data[0], LLMCacheControlData)
