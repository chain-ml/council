from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import yaml
from council.utils import DataObject, DataObjectSpecBase
from council.utils.parameter import Undefined


class LLMProviders(str, Enum):
    """Supported LLM providers."""

    OpenAI = "openAISpec"
    Azure = "azureSpec"
    Anthropic = "anthropicSpec"
    Gemini = "googleGeminiSpec"
    Ollama = "ollamaSpec"
    Groq = "groqSpec"

    @staticmethod
    def all() -> List[LLMProviders]:
        return list(LLMProviders.__members__.values())


class LLMProvider:
    def __init__(self, name: str, description: str, specs: Dict[str, Any], kind: LLMProviders) -> None:
        self.name = name
        self.description = description
        self._specs = specs
        self._kind = kind

    @property
    def kind(self) -> LLMProviders:
        return self._kind

    def is_of_kind(self, kind: LLMProviders) -> bool:
        return self._kind == kind

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMProvider:
        name = values.get("name", "")
        description = values.get("description", "")

        provider_specs: Mapping[LLMProviders, Optional[Dict[str, Any]]] = {
            provider: values.get(provider) for provider in LLMProviders.all()
        }

        for provider, spec in provider_specs.items():
            if spec is not None:
                return LLMProvider(name, description, spec, provider)

        raise ValueError("Unsupported model provider")

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name, "description": self.description}

        for provider in LLMProviders.all():
            if self.is_of_kind(provider):
                result[provider] = self._specs
                break

        return result

    def must_get_value(self, key: str) -> Any:
        return self.get_value(key=key, required=True)

    def get_value(self, key: str, required: bool = False, default: Optional[Any] = Undefined()) -> Optional[Any]:
        maybe_value = self._specs.get(key, None)
        if maybe_value is None:
            if not isinstance(default, Undefined):
                return default

        if isinstance(maybe_value, dict):
            default_value: Optional[str] = maybe_value.get("default", None)
            env_var_name: Optional[str] = maybe_value.get("fromEnvVar", None)
            if env_var_name is not None:
                maybe_value = os.environ.get(env_var_name, default_value)

        if maybe_value is None and required:
            raise Exception(f"LLMProvider {self.name} - A required key {key} is missing.")
        return maybe_value

    def __str__(self) -> str:
        return f"{self._kind}: {self.name} ({self.description})"


class LLMConfigSpec(DataObjectSpecBase):
    def __init__(
        self, description: str, provider: LLMProvider, fallback: Optional[LLMProvider], parameters: Dict[str, Any]
    ) -> None:
        self.description = description
        self.provider = provider
        self.parameters = parameters
        self.fallback_provider = fallback

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMConfigSpec:
        description = values.get("description", "")
        parameters = values.get("parameters", {})
        fallback_spec: Optional[Dict[str, Any]] = values.get("fallbackProvider", None)
        fallback = LLMProvider.from_dict(fallback_spec) if fallback_spec is not None else None
        provider = LLMProvider.from_dict(values["provider"])
        if provider is None:
            raise ValueError("provider needs to be defined.")

        return LLMConfigSpec(description, provider, fallback, parameters)

    def to_dict(self) -> Dict[str, Any]:
        result = {"description": self.description, "provider": self.provider, "parameters": self.parameters}
        if self.fallback_provider is not None:
            result["fallback_provider"] = self.fallback_provider
        return result

    def __str__(self) -> str:
        return f"{self.description}"

    def check_provider(self, provider: LLMProviders) -> None:
        if not self.provider.is_of_kind(provider):
            raise ValueError(f"Invalid LLM provider, actual {self.provider}, expected {provider}")


class LLMConfigObject(DataObject[LLMConfigSpec]):
    """
    Helper class to instantiate an LLM from a YAML file
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMConfigObject:
        return super()._from_dict(LLMConfigSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMConfigObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMConfig")
            return LLMConfigObject.from_dict(values)
