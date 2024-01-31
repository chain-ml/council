"""This package provides clients to use various LLMs"""
from typing import Optional
from ..utils import read_env_str


from .llm_config_object import LLMProvider, LLMConfigObject, LLMConfigSpec, LLMProviders
from .llm_answer import llm_property, LLMAnswer, LLMProperty
from .llm_exception import LLMException, LLMCallException, LLMCallTimeoutException, LLMTokenLimitException
from .llm_message import LLMMessageRole, LLMMessage, LLMessageTokenCounterBase
from .llm_base import LLMBase, LLMResult
from .llm_answer import LLMAnswer, LLMParsingException
from .monitored_llm import MonitoredLLM
from .llm_configuration_base import LLMConfigurationBase
from .llm_fallback import LLMFallback

from .openai_chat_completions_llm import OpenAIChatCompletionsModel
from .openai_token_counter import OpenAITokenCounter

from .azure_llm_configuration import AzureLLMConfiguration
from .azure_llm import AzureLLM

from .openai_llm_configuration import OpenAILLMConfiguration
from .openai_llm import OpenAILLM

from .anthropic_llm_configuration import AnthropicLLMConfiguration
from .anthropic_llm import AnthropicLLM


def get_default_llm(max_retries: Optional[int] = None) -> LLMBase:
    provider = read_env_str("COUNCIL_DEFAULT_LLM_PROVIDER", default=LLMProviders.OpenAI).unwrap()
    provider = provider.lower() + "spec"
    llm: Optional[LLMBase] = None

    if provider == LLMProviders.OpenAI.lower():
        llm = OpenAILLM.from_env()
    elif provider == LLMProviders.Azure.lower():
        llm = AzureLLM.from_env()
    elif provider == LLMProviders.Anthropic.lower():
        llm = AnthropicLLM.from_env()

    if llm is None:
        raise ValueError(f"Provider {provider} not supported by council.")

    if max_retries is not None and max_retries > 0:
        return LLMFallback(llm=llm, fallback=llm, retry_before_fallback=max_retries - 1)

    return llm


def get_llm_from_config(filename: str) -> LLMBase:
    llm_config = LLMConfigObject.from_yaml(filename)

    llm = _build_llm(llm_config)
    fallback_provider = llm_config.spec.fallback_provider
    if fallback_provider is not None:
        llm_config.spec.provider = fallback_provider
        llm_with_fallback = _build_llm(llm_config)
        return LLMFallback(llm, llm_with_fallback)
    return llm


def _build_llm(llm_config: LLMConfigObject) -> LLMBase:
    provider = llm_config.spec.provider
    if provider.is_of_kind(LLMProviders.Azure):
        return AzureLLM.from_config(llm_config)
    elif provider.is_of_kind(LLMProviders.OpenAI):
        return OpenAILLM.from_config(llm_config)
    elif provider.is_of_kind(LLMProviders.OpenAI):
        return OpenAILLM.from_config(llm_config)

    raise ValueError(f"Provider `{provider.kind}` not supported by Council")
