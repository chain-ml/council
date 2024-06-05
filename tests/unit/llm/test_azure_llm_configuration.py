import unittest
from council.llm import AzureChatGptConfiguration
from council.utils import OsEnviron, ParameterValueException


class TestAzureLLMConfiguration(unittest.TestCase):
    def test_from_env_value(self):
        with (
            OsEnviron("AZURE_LLM_API_KEY", "aKeY"),
            OsEnviron("AZURE_LLM_API_BASE", "council"),
            OsEnviron("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4"),
        ):
            config = AzureChatGptConfiguration.from_env()
            self.assertEqual("aKeY", config.api_key.value)
            self.assertEqual("council", config.api_base.value)
            self.assertEqual("gpt-4", config.deployment_name.value)
            self.assertEqual(30, config.timeout.value)

    def test_from_env_value_wo_deployment_name(self):
        with (
            OsEnviron("AZURE_LLM_API_KEY", "aKeY"),
            OsEnviron("AZURE_LLM_API_BASE", "council"),
        ):
            config = AzureChatGptConfiguration.from_env(deployment_name="gpt-4")
            self.assertEqual("aKeY", config.api_key.value)
            self.assertEqual("council", config.api_base.value)
            self.assertEqual("gpt-4", config.deployment_name.value)
            self.assertEqual(30, config.timeout.value)

    def test_from_env_override_default_value(self):
        with (
            OsEnviron("AZURE_LLM_API_KEY", "aKeY"),
            OsEnviron("AZURE_LLM_API_BASE", "council"),
            OsEnviron("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4"),
            OsEnviron("AZURE_LLM_TIMEOUT", "90"),
        ):
            config = AzureChatGptConfiguration.from_env()
            self.assertEqual(90, config.timeout.value)

    def test_default(self):
        config = AzureChatGptConfiguration(api_key="aKeY", api_base="council", deployment_name="gpt-4")
        self.assertEqual("aKeY", config.api_key.value)
        self.assertEqual("council", config.api_base.value)
        self.assertEqual("gpt-4", config.deployment_name.value)

        self.assertEqual(30, config.timeout.value)
        self.assertEqual(0.0, config.temperature.value)
        self.assertEqual(1, config.n.value)
        self.assertTrue(config.top_p.is_none())
        self.assertTrue(config.frequency_penalty.is_none())
        self.assertTrue(config.presence_penalty.is_none())

    def test_invalid(self):
        with self.assertRaises(ParameterValueException):
            _ = AzureChatGptConfiguration(api_key=" ", api_base="council", deployment_name="gpt-4")
        with self.assertRaises(ParameterValueException):
            _ = AzureChatGptConfiguration(api_key="aKeY", api_base=" ", deployment_name="gpt-4")
        with self.assertRaises(ParameterValueException):
            _ = AzureChatGptConfiguration(api_key="aKeY", api_base="council", deployment_name=" ")
