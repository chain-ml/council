from council import OpenAILLM, AzureLLM, AnthropicLLM
from council.llm import LLMBase
from council.utils import read_env_str

from enum import Enum


class Providers(str, Enum):
    OpenAI = "OpenAI"
    Azure = "Azure"
    Anthropic = "Anthropic"


def get_test_default_llm() -> LLMBase:
    provider = read_env_str("COUNCIL_DEFAULT_LLM_PROVIDER", default=Providers.OpenAI).unwrap()
    provider = provider.lower()
    if provider == Providers.OpenAI.lower():
        return OpenAILLM.from_env()

    if provider == Providers.Azure.lower():
        return AzureLLM.from_env()

    if provider == Providers.Anthropic.lower():
        return AnthropicLLM.from_env()

    raise Exception(f"Provider {provider} not found")
