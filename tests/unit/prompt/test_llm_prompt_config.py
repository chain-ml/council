import unittest

import yaml

from council.prompt import (
    LLMPromptConfigObject,
    LLMPromptConfigSpec,
    LLMStructuredPromptConfigObject,
    LLMStructuredPromptConfigSpec,
    XMLPromptFormatter,
    MarkdownPromptFormatter,
)

from tests import get_data_filename
from .. import LLMPrompts, LLMStructuredPrompts


class TestLLMPromptConfig(unittest.TestCase):
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


class TestLLMStructuredPrompt(unittest.TestCase):
    @staticmethod
    def load_sample_prompt() -> LLMStructuredPromptConfigObject:
        filename = get_data_filename(LLMStructuredPrompts.sample)
        return LLMStructuredPromptConfigObject.from_yaml(filename)

    def test_structured_prompt_from_yaml(self):
        actual = self.load_sample_prompt()

        assert isinstance(actual, LLMStructuredPromptConfigObject)
        assert actual.kind == "LLMStructuredPrompt"

    def test_xml_structured_prompt(self):
        prompt = self.load_sample_prompt()
        prompt.set_formatter(XMLPromptFormatter())

        assert (
            prompt.get_system_prompt_template("default")
            == """<role>
  You are a helpful assistant.
  <instructions>
    Answer user questions.
  </instructions>
  <rules>
    Here are rules to follow.
    <rule_1>
      Be nice.
    </rule_1>
    <rule_2>
      Be specific.
    </rule_2>
  </rules>
</role>
<context>
  The user is asking about programming concepts.
</context>
<response_template>
  Provide the answer in simple terms.
</response_template>"""
        )

        assert (
            prompt.get_user_prompt_template("default")
            == """<question>
  Explain what is object-oriented programming.
</question>"""
        )

    def test_md_structured_prompt(self):
        prompt = self.load_sample_prompt()
        prompt.set_formatter(MarkdownPromptFormatter())

        assert (
            prompt.get_system_prompt_template("default")
            == """# Role
You are a helpful assistant.
## Instructions
Answer user questions.
## Rules
Here are rules to follow.
### Rule 1
Be nice.
### Rule 2
Be specific.
# Context
The user is asking about programming concepts.
# Response template
Provide the answer in simple terms."""
        )

        assert (
            prompt.get_user_prompt_template("default")
            == """# Question
Explain what is object-oriented programming."""
        )

    def test_parse_no_system(self):
        prompt_config_spec = """
        spec:
          user:
            - model: default
              sections:
                - name: user
                  content: |
                    User prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMStructuredPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "System prompt(s) must be defined"

    def test_parse_no_user(self):
        prompt_config_spec = """
        spec:
          system:
            - model: default
              sections:
                - name: system
                  content: |
                    System prompt template
        """
        values = yaml.safe_load(prompt_config_spec)
        _ = LLMStructuredPromptConfigSpec.from_dict(values["spec"])
