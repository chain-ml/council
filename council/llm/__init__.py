"""This package provides clients to use various LLMs."""

from typing import Optional, Type

from ..utils import read_env_str

from .llm_config_object import LLMProvider, LLMConfigObject, LLMConfigSpec, LLMProviders
from .llm_answer import llm_property, LLMAnswer, LLMProperty, LLMParsingException
from .llm_exception import LLMException, LLMCallException, LLMCallTimeoutException, LLMTokenLimitException
from .llm_message import LLMMessageRole, LLMMessage, LLMMessageTokenCounterBase
from .llm_base import LLMBase, LLMResult, LLMConfigurationBase
from .llm_cost import (
    LLMCostCard,
    LLMConsumptionCalculatorBase,
    DefaultLLMConsumptionCalculator,
    DefaultLLMConsumptionCalculatorHelper,
    TokenKind,
    LLMCostManagerSpec,
    LLMCostManagerObject,
)
from .llm_fallback import LLMFallback
from .llm_middleware import (
    LLMRequest,
    LLMResponse,
    LLMMiddleware,
    LLMMiddlewareChain,
    LLMRetryMiddleware,
    LLMLoggingStrategy,
    LLMLoggingMiddleware,
    LLMFileLoggingMiddleware,
    LLMCachingMiddleware,
    ExecuteLLMRequest,
)
from .llm_response_parser import (
    EchoResponseParser,
    StringResponseParser,
    CodeBlocksResponseParser,
    JSONBlockResponseParser,
    JSONResponseParser,
    YAMLBlockResponseParser,
    YAMLResponseParser,
)
from .llm_function import LLMFunction, LLMFunctionResponse, LLMFunctionError, FunctionOutOfRetryError
from .llm_function_with_prompt import LLMFunctionWithPrompt
from .monitored_llm import MonitoredLLM

from .providers import (
    _build_llm,
    _PROVIDER_TO_LLM,
    AzureLLM,
    AzureChatGPTConfiguration,
    OpenAILLM,
    OpenAIChatGPTConfiguration,
    AnthropicLLM,
    AnthropicLLMConfiguration,
    GeminiLLM,
    GeminiLLMConfiguration,
    GroqLLM,
    GroqLLMConfiguration,
    OllamaLLM,
    OllamaLLMConfiguration,
)


def get_default_llm(max_retries: Optional[int] = None) -> LLMBase:
    """Get default LLM based on `COUNCIL_DEFAULT_LLM_PROVIDER` env variable."""
    provider_str = read_env_str("COUNCIL_DEFAULT_LLM_PROVIDER", default=LLMProviders.OpenAI).unwrap()
    provider_str = provider_str.lower() + "spec"

    llm_class: Optional[Type[LLMBase]] = next(
        (llm_class for provider_enum, llm_class in _PROVIDER_TO_LLM.items() if provider_str == provider_enum.lower()),
        None,
    )

    if llm_class is None:
        raise ValueError(f"Provider {provider_str} not supported by Council.")

    llm = llm_class.from_env()

    if max_retries is not None and max_retries > 0:
        return LLMFallback(llm=llm, fallback=llm, retry_before_fallback=max_retries - 1)

    return llm


def get_llm_from_config(filename: str) -> LLMBase:
    """Get LLM from a yaml LLMConfigObject file."""
    llm_config = LLMConfigObject.from_yaml(filename)
    return get_llm_from_config_obj(llm_config)


def get_llm_from_config_obj(llm_config: LLMConfigObject):
    llm = _build_llm(llm_config)
    fallback_provider = llm_config.spec.fallback_provider
    if fallback_provider is not None:
        llm_config.spec.provider = fallback_provider
        llm_with_fallback = _build_llm(llm_config)
        return LLMFallback(llm, llm_with_fallback)
    return llm
