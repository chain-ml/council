from council import OpenAILLM, AzureLLM, AnthropicLLM, GeminiLLM
from council.llm import (
    get_llm_from_config,
    LLMFallback,
    OpenAIChatGPTConfiguration,
    OllamaLLM,
    OllamaLLMConfiguration,
    GroqLLM,
    GroqLLMConfiguration,
    LLMConfigObject,
)
from council.utils import OsEnviron

from tests import get_data_filename
from .. import LLMModels


def test_openai_from_yaml():
    filename = get_data_filename(LLMModels.OpenAI)

    with OsEnviron("OPENAI_API_KEY", "sk-key"), OsEnviron("OPENAI_API_HOST", "https://openai.com"):
        actual = LLMConfigObject.from_yaml(filename)
        assert actual.spec.provider.name == "CML-OpenAI"

        llm = OpenAILLM.from_config(actual)
        assert isinstance(llm, OpenAILLM)

        assert isinstance(llm.configuration, OpenAIChatGPTConfiguration)
        config: OpenAIChatGPTConfiguration = llm.configuration
        assert config.temperature == 0.5
        assert config.n == 3
        assert config.api_host == "https://openai.com"

        llm = get_llm_from_config(filename)
        assert isinstance(llm, OpenAILLM)


def test_azure_from_yaml():
    filename = get_data_filename(LLMModels.Azure)

    with OsEnviron("AZURE_LLM_API_KEY", "abcd"), OsEnviron("AZURE_LLM_API_BASE", "https://chainml"):
        actual = LLMConfigObject.from_yaml(filename)
        assert actual.spec.provider.name == "CML-Azure"

        llm = AzureLLM.from_config(actual)
        assert isinstance(llm, AzureLLM)
        llm = get_llm_from_config(filename)
        assert isinstance(llm, AzureLLM)


def test_anthropic_from_yaml():
    filename = get_data_filename(LLMModels.Anthropic)

    with OsEnviron("ANTHROPIC_API_KEY", "sk-key"):
        actual = LLMConfigObject.from_yaml(filename)
        llm = AnthropicLLM.from_config(actual)
        assert isinstance(llm, AnthropicLLM)
        assert llm.configuration.top_k.value == 8

        llm = get_llm_from_config(filename)
        assert isinstance(llm, AnthropicLLM)


def test_gemini_from_yaml():
    filename = get_data_filename(LLMModels.Gemini)

    with OsEnviron("GEMINI_API_KEY", "a-key"):
        actual = LLMConfigObject.from_yaml(filename)
        llm = GeminiLLM.from_config(actual)
        assert isinstance(llm, GeminiLLM)
        assert llm.configuration.top_k.value == 8

        llm = get_llm_from_config(filename)
        assert isinstance(llm, GeminiLLM)


def test_azure_with_openai_fallback_from_yaml():
    filename = get_data_filename(LLMModels.AzureWithFallback)

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


def test_ollama_from_yaml():
    filename = get_data_filename(LLMModels.Ollama)

    actual = LLMConfigObject.from_yaml(filename)
    llm = OllamaLLM.from_config(actual)
    assert isinstance(llm, OllamaLLM)

    config: OllamaLLMConfiguration = llm.configuration
    assert config.model.value == "llama3.2"
    assert config.keep_alive_value == 300
    assert config.json_mode.value
    assert config.temperature.value == 0.8
    assert config.repeat_penalty.value == 0.7
    assert config.top_p.value == 0.2
    assert config.seed.value == 42
    assert config.mirostat_eta.value == 0.314
    assert config.num_ctx.value == 4096
    assert config.num_predict.value == 512

    llm = get_llm_from_config(filename)
    assert isinstance(llm, OllamaLLM)


def test_groq_from_yaml():
    filename = get_data_filename(LLMModels.Groq)

    with OsEnviron("GROQ_API_KEY", "a-key"):
        actual = LLMConfigObject.from_yaml(filename)
        llm = GroqLLM.from_config(actual)
        assert isinstance(llm, GroqLLM)
        config: GroqLLMConfiguration = llm.configuration
        assert config.model.value == "llama-3.2-1b-preview"
        assert config.frequency_penalty.value == 0.7
        assert config.max_tokens.value == 24
        assert config.presence_penalty.value == -0.4
        assert config.seed.value == 42
        assert config.temperature.value == 0.5
        assert config.top_p.value == 0.1

        llm = get_llm_from_config(filename)
        assert isinstance(llm, GroqLLM)
