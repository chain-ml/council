import httpx

from typing import Any

from . import OpenAIChatCompletionsModel
from .openai_llm_configuration import OpenAILLMConfiguration


class OpenAIChatCompletionsModelProvider:
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: OpenAILLMConfiguration

    def __init__(self, config: OpenAILLMConfiguration):
        self.config = config

    def post_request(self, payload: dict[str, Any]) -> httpx.Response:
        uri = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": self.config.authorization, "Content-Type": "application/json"}

        with httpx.Client() as client:
            client.timeout.read = self.config.timeout
            return client.post(url=uri, headers=headers, json=payload)


class OpenAILLM(OpenAIChatCompletionsModel):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: OpenAILLMConfiguration

    def __init__(self, config: OpenAILLMConfiguration):
        super().__init__(config, OpenAIChatCompletionsModelProvider(config).post_request)

    @staticmethod
    def from_env() -> "OpenAILLM":
        config: OpenAILLMConfiguration = OpenAILLMConfiguration.from_env()
        return OpenAILLM(config)
