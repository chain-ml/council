from __future__ import annotations

from typing import Any, Optional

import httpx
from httpx import HTTPStatusError, TimeoutException

from ...llm_config_object import LLMConfigObject
from ...llm_exception import LLMCallException, LLMCallTimeoutException
from .openai_chat_completions_llm import OpenAIChatCompletionsModel
from .openai_chat_gpt_configuration import OpenAIChatGPTConfiguration
from .openai_token_counter import OpenAITokenCounter


class OpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(self, config: OpenAIChatGPTConfiguration, name: Optional[str] = None) -> None:
        self.config = config
        bearer = f"Bearer {config.api_key.unwrap()}"
        self._headers = {"Authorization": bearer, "Content-Type": "application/json"}
        self._name = name

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        """
        Posts a request to the OpenAI chat completions endpoint.
        """
        uri = self.config.api_host.unwrap() + "/v1/chat/completions"

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

    def __init__(self, config: OpenAIChatGPTConfiguration, name: Optional[str] = None) -> None:
        name = name or f"{self.__class__.__name__}"
        super().__init__(
            config,
            OpenAIChatCompletionsModelProvider(config, name).post_request,
            token_counter=OpenAITokenCounter.from_model(config.model.unwrap_or("")),
            name=name,
        )

    @staticmethod
    def from_env(model: Optional[str] = None, api_host: Optional[str] = None) -> OpenAILLM:
        config: OpenAIChatGPTConfiguration = OpenAIChatGPTConfiguration.from_env(model=model, api_host=api_host)
        return OpenAILLM(config)

    @classmethod
    def from_config(cls, config_object: LLMConfigObject) -> OpenAILLM:
        config = OpenAIChatGPTConfiguration.from_spec(config_object.spec)
        return OpenAILLM(config=config, name=config_object.metadata.name)
