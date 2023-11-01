from typing import Optional

from council import OpenAILLM, AzureLLM, AnthropicLLM
from council.llm import LLMBase, LLMFallback
from council.utils import read_env_str

from enum import Enum


class Providers(str, Enum):
    OpenAI = "OpenAI"
    Azure = "Azure"
    Anthropic = "Anthropic"


def get_test_default_llm(max_retries: Optional[int] = None) -> LLMBase:
    provider = read_env_str("COUNCIL_DEFAULT_LLM_PROVIDER", default=Providers.OpenAI).unwrap()
    provider = provider.lower()
    llm: Optional[LLMBase] = None
    if provider == Providers.OpenAI.lower():
        llm = OpenAILLM.from_env()
    elif provider == Providers.Azure.lower():
        llm = AzureLLM.from_env()
    elif provider == Providers.Anthropic.lower():
        llm = AnthropicLLM.from_env()

    if llm is None:
        raise Exception(f"Provider {provider} not found")

    if max_retries is not None and max_retries > 0:
        return LLMFallback(llm=llm, fallback=llm, retry_before_fallback=max_retries - 1)

    return llm
