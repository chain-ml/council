from council import OpenAILLM, AzureLLM, AnthropicLLM
from council.llm import get_llm_from_config, LLMFallback
from council.llm.llm_config_object import LLMConfigObject
from council.utils import OsEnviron

from .. import get_data_filename, LLModels


def test_openai_from_yaml():
    filename = get_data_filename(LLModels.OpenAI)

    with OsEnviron("OPENAI_API_KEY", "sk-key"):
        actual = LLMConfigObject.from_yaml(filename)
        assert actual.spec.provider.name == "CML-OpenAI"

        llm = OpenAILLM.from_config(actual)
        assert isinstance(llm, OpenAILLM)
        assert llm.config.temperature.value == 0.5
        assert llm.config.n.value == 3

        llm = get_llm_from_config(filename)
        assert isinstance(llm, OpenAILLM)


def test_azure_from_yaml():
    filename = get_data_filename(LLModels.Azure)

    with OsEnviron("AZURE_LLM_API_KEY", "abcd"), OsEnviron("AZURE_LLM_API_BASE", "https://chainml"):
        actual = LLMConfigObject.from_yaml(filename)
        assert actual.spec.provider.name == "CML-Azure"

        llm = AzureLLM.from_config(actual)
        assert isinstance(llm, AzureLLM)
        llm = get_llm_from_config(filename)
        assert isinstance(llm, AzureLLM)


def test_anthropic_from_yaml():
    filename = get_data_filename(LLModels.Anthropic)

    with OsEnviron("ANTHROPIC_API_KEY", "sk-key"):
        actual = LLMConfigObject.from_yaml(filename)
        llm = AnthropicLLM.from_config(actual)
        assert isinstance(llm, AnthropicLLM)
        assert llm.config.top_k.value == 8

        llm = get_llm_from_config(filename)
        assert isinstance(llm, AnthropicLLM)


def test_azure_with_openai_fallback_from_yaml():
    filename = get_data_filename(LLModels.AzureWithFallback)

    with (
        OsEnviron("OPENAI_API_KEY", "sk-key"),
        OsEnviron("OPENAI_LLM_MODEL", "gpt-not-default"),
        OsEnviron("AZURE_LLM_API_KEY", "abcd"),
        OsEnviron("AZURE_LLM_API_BASE", "https://chainml"),
    ):
        llm = get_llm_from_config(filename)
        assert isinstance(llm, LLMFallback)
        assert isinstance(llm.llm, AzureLLM)
        assert isinstance(llm.fallback, OpenAILLM)
