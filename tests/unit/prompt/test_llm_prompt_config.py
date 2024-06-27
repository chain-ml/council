import unittest

import yaml

from council.prompt import LLMPromptConfigObject, LLMPromptConfigSpec

from tests import get_data_filename
from .. import LLMPrompts


class TestLLMFallBack(unittest.TestCase):
    def test_llm_prompt_from_yaml(self):
        filename = get_data_filename(LLMPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        assert isinstance(actual, LLMPromptConfigObject)
        assert actual.kind == "LLMPrompt"

    def test_llm_prompt_templates(self):
        filename = get_data_filename(LLMPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        system_prompt_gpt4o = actual.get_system_prompt_template("gpt-4o")
        assert system_prompt_gpt4o.rstrip("\n") == "System prompt template specific for gpt-4o"
        system_prompt_gpt35 = actual.get_system_prompt_template("gpt-3.5-turbo")
        assert system_prompt_gpt35.rstrip("\n") == "System prompt template for gpt-3.5-turbo and other gpt models"
        system_prompt_gpt = actual.get_system_prompt_template("gpt-4-turbo-preview")
        assert system_prompt_gpt.rstrip("\n") == "System prompt template for gpt-3.5-turbo and other gpt models"
        with self.assertRaises(ValueError) as e:
            _ = actual.get_system_prompt_template("claude-3-opus-20240229")
        assert str(e.exception) == "No prompt template for a given model `claude-3-opus-20240229` nor a default one"

        user_prompt_gpt4_turbo = actual.get_user_prompt_template("gpt-4-turbo-preview")
        assert user_prompt_gpt4_turbo.rstrip("\n") == "User prompt template for gpt-4-turbo-preview"
        user_prompt_gpt4o = actual.get_user_prompt_template("gpt-4o")
        assert user_prompt_gpt4o.rstrip("\n") == "User prompt template for default model"
        user_prompt_claude = actual.get_user_prompt_template("claude-3-opus-20240229")
        assert user_prompt_claude.rstrip("\n") == "User prompt template for default model"

    def test_parse_no_system(self):
        prompt_config_spec = """
        spec:
          user:
            - model: gpt-4o
              template: |
                User prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "System prompt(s) must be defined"

    def test_parse_no_user(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
              template: |
                System prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        _ = LLMPromptConfigSpec.from_dict(values["spec"])

    def test_parse_no_template(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "`template` must be defined"

    def test_parse_no_model_model_family(self):
        prompt_config_spec = """
        spec:
          system:
            - template: template
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "At least one of `model` or `model-family` must be defined"

    def test_no_compliant(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
              model-family: claude
              template: |
                System prompt template specific for gpt-4o or claude models
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception).startswith("model `gpt-4o` and model-family `claude` are not compliant")
