"""

The `__init__` module of the `llm` package provides a high-level interface to initialize default Large Language Models (LLMs) seamlessly with environment-specific configurations or load them from an external configuration file. It also defines utility functions to construct specific instances of LLM providers such as OpenAI, Azure, or Anthropic. The module contains functions for retrieving a default LLM instance with optional retry strategies, as well as for creating a fully configured LLM instance based on YAML configuration files.

Functions:
    - get_default_llm(max_retries): Returns a default instance of LLMBase according to the configured provider. The provider is determined by the environment variable 'COUNCIL_DEFAULT_LLM_PROVIDER'. If `max_retries` is specified and greater than zero, a retry mechanism is applied to the LLM where fallback attempts are made upon failure.
    - get_llm_from_config(filename): Constructs an LLMBase instance using the configurations specified in the given YAML file. If a fallback provider is specified in the configuration, a fallback mechanism is included.
    - _build_llm(llm_config): Helper function used internally to create an instance of an LLM provider based on the provided LLMConfigObject. Raises a ValueError if the provider type is not supported by the Council.


"""

from typing import Optional
from ..utils import read_env_str


from .llm_config_object import LLMProvider, LLMConfigObject, LLMConfigSpec, LLMProviders
from .llm_answer import llm_property, LLMAnswer, LLMProperty, LLMParsingException
from .llm_exception import LLMException, LLMCallException, LLMCallTimeoutException, LLMTokenLimitException
from .llm_message import LLMMessageRole, LLMMessage, LLMessageTokenCounterBase
from .llm_base import LLMBase, LLMResult
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
    """
    Retrieves the default language learning model (LLM) based on the configured environment variable.
    This function determines the appropriate LLM provider to use by reading the 'COUNCIL_DEFAULT_LLM_PROVIDER' environment variable.
    The supported providers are 'OpenAI', 'Azure', and 'Anthropic' and are case-insensitive. Each provider has a corresponding method (e.g., 'from_env') used to instantiate their respective LLM object.
    If no provider is set in the environment variable, 'OpenAI' is used as the default.
    Depending on the specified number of retries, the function also provides a fallback mechanism in case the primary LLM encounters issues. If 'max_retries' is set and greater than zero, a 'LLMFallback' object is created, wrapping the primary LLM and providing retry functionality.
    
    Args:
        max_retries (Optional[int]):
             The maximum number of retries before fallback. If None or not provided, no retry logic is applied.
    
    Returns:
        (LLMBase):
             An instance of the requested LLM provider or a fallback wrapper with retry logic.
    
    Raises:
        ValueError:
             If the provider specified in the environment variable is not supported.
        

    """
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
    """
    Creates an LLM (Large Language Model) object from a given configuration file.
    This function takes the filename of a configuration file in YAML format, constructs an LLM object based on the configuration,
    and returns it. If a fallback provider is specified in the configuration, it will create another instance of the LLM with the
    fallback provider and wrap both in an LLMFallback object to handle retries and fallback logic.
    
    Args:
        filename (str):
             The path to the YAML configuration file used to set up the LLM object.
    
    Returns:
        (LLMBase):
             An instance of an LLM object or an LLMFallback object if a fallback provider is specified in the configuration.
    
    Raises:
        ValueError:
             If the provider specified in the configuration is not supported.
        

    """
    llm_config = LLMConfigObject.from_yaml(filename)

    llm = _build_llm(llm_config)
    fallback_provider = llm_config.spec.fallback_provider
    if fallback_provider is not None:
        llm_config.spec.provider = fallback_provider
        llm_with_fallback = _build_llm(llm_config)
        return LLMFallback(llm, llm_with_fallback)
    return llm


def _build_llm(llm_config: LLMConfigObject) -> LLMBase:
    """
    Builds a Large Language Model (LLM) instance based on provided configuration.
    This function takes a configuration object for a Large Language Model and initializes
    the appropriate LLM instance based on the specified provider. It supports Azure, OpenAI,
    and Anthropic as providers. If the provider is not recognized, it raises a ValueError.
    
    Args:
        llm_config (LLMConfigObject):
             An instance of the configuration object that contains
            the specification for the LLM provider and other relevant configurations.
    
    Returns:
        (LLMBase):
             An instance of the LLM subclass corresponding to the provider specified
            in the configuration object.
    
    Raises:
        ValueError:
             If the LLM provider specified in the configuration is not supported.

    """
    provider = llm_config.spec.provider
    if provider.is_of_kind(LLMProviders.Azure):
        return AzureLLM.from_config(llm_config)
    elif provider.is_of_kind(LLMProviders.OpenAI):
        return OpenAILLM.from_config(llm_config)
    elif provider.is_of_kind(LLMProviders.Anthropic):
        return AnthropicLLM.from_config(llm_config)

    raise ValueError(f"Provider `{provider.kind}` not supported by Council")
