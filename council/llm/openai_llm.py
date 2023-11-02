import httpx

from typing import Any, Optional

from httpx import TimeoutException, HTTPStatusError

from . import OpenAIChatCompletionsModel, OpenAITokenCounter, LLMCallTimeoutException, LLMCallException
from .openai_llm_configuration import OpenAILLMConfiguration


class OpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: OpenAILLMConfiguration

    def __init__(self, config: OpenAILLMConfiguration):
        self.config = config
        bearer = f"Bearer {config.api_key.unwrap()}"
        self._headers = {"Authorization": bearer, "Content-Type": "application/json"}

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        uri = "https://api.openai.com/v1/chat/completions"

        timeout = self.config.timeout.unwrap()
        try:
            with httpx.Client() as client:
                client.timeout.read = timeout
                return client.post(url=uri, headers=self._headers, json=payload)
        except TimeoutException as e:
            raise LLMCallTimeoutException(timeout) from e
        except HTTPStatusError as e:
            raise LLMCallException(code=e.response.status_code, error=e.response.text) from e


class OpenAILLM(OpenAIChatCompletionsModel):
    """
    Represents an OpenAI large language model hosted on OpenAI.
    """

    config: OpenAILLMConfiguration

    def __init__(self, config: OpenAILLMConfiguration):
        super().__init__(
            config,
            OpenAIChatCompletionsModelProvider(config).post_request,
            token_counter=OpenAITokenCounter.from_model(config.model.unwrap_or("")),
        )

    @staticmethod
    def from_env(model: Optional[str] = None) -> "OpenAILLM":
        config: OpenAILLMConfiguration = OpenAILLMConfiguration.from_env(model=model)
        return OpenAILLM(config)
