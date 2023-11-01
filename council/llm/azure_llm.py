from typing import Any

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

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        uri = f"{self.config.api_base}/openai/deployments/{self.config.deployment_name}/chat/completions"
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        params = {"api-version": self.config.api_version}

        try:
            with httpx.Client() as client:
                client.timeout.read = self.config.timeout
                return client.post(url=uri, headers=headers, params=params, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(self.config.timeout) from e
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
    def from_env() -> "AzureLLM":
        config: AzureLLMConfiguration = AzureLLMConfiguration.from_env()
        return AzureLLM(config)
