import unittest

import yaml

from council.prompt import XMLPromptFormatter, XMLPromptSpec, XMLSection

from tests import get_data_filename
from .. import XMLPrompts


class TestXMLPromptFormatter(unittest.TestCase):
    def test_xml_section(self):
        section = XMLSection(name="test", content="content")
        assert section.to_xml() == "<test>\ncontent\n</test>"

    def test_xml_section_snake_case(self):
        section = XMLSection(name="  Complex nAmE    ", content="Complex Content 123")
        assert section.to_xml() == "<complex_name>\nComplex Content 123\n</complex_name>"

    def test_xml_prompt_from_yaml(self):
        filename = get_data_filename(XMLPrompts.sample)
        actual = XMLPromptFormatter.from_yaml(filename)

        assert isinstance(actual, XMLPromptFormatter)
        assert actual.kind == "XMLPrompt"

    def test_sample_xml_prompt(self):
        filename = get_data_filename(XMLPrompts.sample)
        actual = XMLPromptFormatter.from_yaml(filename)

        assert (
            actual.get_system_prompt_template()
            == """<role>
You are a helpful assistant.
</role>
<context>
The user is asking about programming concepts.
</context>"""
        )

        assert (
            actual.get_user_prompt_template()
            == """<question>
Explain what is object-oriented programming.
</question>
<response_template>
Provide the answer in simple terms.
</response_template>"""
        )

    def test_parse_no_system(self):
        prompt_config_spec = """
        spec:
          user:
            - name: user
              content: |
                User prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLPromptSpec.from_dict(values["spec"])
        assert str(e.exception) == "System section(s) must be defined"

    def test_parse_no_user(self):
        prompt_config_spec = """
        spec:
          system:
            - name: system
              content: |
                System prompt template
        """
        values = yaml.safe_load(prompt_config_spec)
        _ = XMLPromptSpec.from_dict(values["spec"])

    def test_parse_no_template(self):
        prompt_config_spec = """
        spec:
          system:
            - name: system
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLPromptSpec.from_dict(values["spec"])
        assert str(e.exception) == "Both 'name' and 'content' must be defined"
