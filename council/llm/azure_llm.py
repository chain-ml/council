from typing import Any, Optional

import httpx
from httpx import TimeoutException, HTTPStatusError

from . import OpenAIChatCompletionsModel, LLMCallTimeoutException, LLMCallException
from .azure_llm_configuration import AzureLLMConfiguration


class AzureOpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: AzureLLMConfiguration

    def __init__(self, config: AzureLLMConfiguration):
        self.config = config
        self._uri = (
            f"{self.config.api_base.value}/openai/deployments/{self.config.deployment_name.value}/chat/completions"
        )

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"api-key": self.config.api_key.unwrap(), "Content-Type": "application/json"}
        params = {"api-version": self.config.api_version.value}

        timeout = self.config.timeout.value
        try:
            with httpx.Client() as client:
                client.timeout.read = timeout
                return client.post(url=self._uri, headers=headers, params=params, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text) from e


class AzureLLM(OpenAIChatCompletionsModel):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: AzureLLMConfiguration

    def __init__(self, config: AzureLLMConfiguration):
        super().__init__(config, AzureOpenAIChatCompletionsModelProvider(config).post_request, None)

    @staticmethod
    def from_env(deployment_name: Optional[str] = None) -> "AzureLLM":
        config: AzureLLMConfiguration = AzureLLMConfiguration.from_env(deployment_name)
        return AzureLLM(config)
