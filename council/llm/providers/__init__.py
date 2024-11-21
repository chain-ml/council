from typing import Mapping, Type, Optional

from .anthropic import AnthropicLLM, AnthropicLLMConfiguration
from .gemini import GeminiLLM, GeminiLLMConfiguration
from .groq import GroqLLM, GroqLLMConfiguration
from .ollama import OllamaLLM, OllamaLLMConfiguration
from .openai import AzureLLM, AzureChatGPTConfiguration, OpenAILLM, OpenAIChatGPTConfiguration
from .. import LLMConfigObject, LLMBase, LLMProviders

_PROVIDER_TO_LLM: Mapping[LLMProviders, Type[LLMBase]] = {
    LLMProviders.Azure: AzureLLM,
    LLMProviders.OpenAI: OpenAILLM,
    LLMProviders.Anthropic: AnthropicLLM,
    LLMProviders.Gemini: GeminiLLM,
    LLMProviders.Ollama: OllamaLLM,
    LLMProviders.Groq: GroqLLM,
}


def _build_llm(llm_config: LLMConfigObject) -> LLMBase:
    provider = llm_config.spec.provider

    llm_class: Optional[Type[LLMBase]] = next(
        (llm_class for provider_enum, llm_class in _PROVIDER_TO_LLM.items() if provider.is_of_kind(provider_enum)), None
    )

    if llm_class is None:
        raise ValueError(f"Provider `{provider.kind}` not supported by Council")

    return llm_class.from_config(llm_config)
