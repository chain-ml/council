from __future__ import annotations
from typing import Any, Optional

import httpx
from httpx import TimeoutException, HTTPStatusError

from . import OpenAIChatCompletionsModel, OpenAITokenCounter, LLMCallTimeoutException, LLMCallException
from .llm_config_object import LLMConfigObject, LLMProviders
from .openai_llm_configuration import OpenAILLMConfiguration


class OpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(self, config: OpenAILLMConfiguration, name: Optional[str] = None) -> None:
        self.config = config
        bearer = f"Bearer {config.api_key.unwrap()}"
        self._headers = {"Authorization": bearer, "Content-Type": "application/json"}
        self._name = name

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        uri = "https://api.openai.com/v1/chat/completions"

        timeout = self.config.timeout.unwrap()
        try:
            with httpx.Client(timeout=timeout) as client:
                return client.post(url=uri, headers=self._headers, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout=timeout, llm_name=self._name) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text, llm_name=self._name) from e


class OpenAILLM(OpenAIChatCompletionsModel):
    """
    Represents an OpenAI large language model hosted on OpenAI.
    """

    def __init__(self, config: OpenAILLMConfiguration, name: Optional[str] = None):
        name = name or f"{self.__class__.__name__}"
        super().__init__(
            config,
            OpenAIChatCompletionsModelProvider(config, name).post_request,
            token_counter=OpenAITokenCounter.from_model(config.model.unwrap_or("")),
            name=name,
        )

    @staticmethod
    def from_env(model: Optional[str] = None) -> OpenAILLM:
        config: OpenAILLMConfiguration = OpenAILLMConfiguration.from_env(model=model)
        return OpenAILLM(config)

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> OpenAILLM:
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.OpenAI):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.OpenAI}")
        config = OpenAILLMConfiguration.from_spec(config_object.spec)
        return OpenAILLM(config=config, name=config_object.metadata.name)
