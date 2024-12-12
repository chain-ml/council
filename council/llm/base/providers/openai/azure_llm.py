from __future__ import annotations

from typing import Any, Optional

import httpx
from httpx import HTTPStatusError, TimeoutException

from ...llm_config_object import LLMConfigObject
from ...llm_exception import LLMCallException, LLMCallTimeoutException
from .azure_chat_gpt_configuration import AzureChatGPTConfiguration
from .openai_chat_completions_llm import OpenAIChatCompletionsModel


class AzureOpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(self, config: AzureChatGPTConfiguration, name: Optional[str]) -> None:
        self.config = config
        self._uri = (
            f"{self.config.api_base.value}/openai/deployments/{self.config.deployment_name.value}/chat/completions"
        )
        self._name = name

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"api-key": self.config.api_key.unwrap(), "Content-Type": "application/json"}
        params = {"api-version": self.config.api_version.value}

        timeout = self.config.timeout.value
        try:
            with httpx.Client(timeout=timeout) as client:
                return client.post(url=self._uri, headers=headers, params=params, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout, self._name) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text, llm_name=self._name) from e


class AzureLLM(OpenAIChatCompletionsModel):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(self, config: AzureChatGPTConfiguration, name: Optional[str] = None) -> None:
        name = name or f"{self.__class__.__name__}"
        super().__init__(config, AzureOpenAIChatCompletionsModelProvider(config, name).post_request, None, name)

    @staticmethod
    def from_env(deployment_name: Optional[str] = None) -> AzureLLM:
        config: AzureChatGPTConfiguration = AzureChatGPTConfiguration.from_env(deployment_name)
        return AzureLLM(config, deployment_name)

    @classmethod
    def from_config(cls, config_object: LLMConfigObject) -> AzureLLM:
        config = AzureChatGPTConfiguration.from_spec(config_object.spec)
        return AzureLLM(config=config, name=config_object.metadata.name)
