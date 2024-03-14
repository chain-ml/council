from __future__ import annotations

from typing import Any, Sequence, Optional, List

from anthropic import Anthropic, APITimeoutError, APIStatusError

from council.contexts import LLMContext, Consumption
from council.llm import (
    LLMBase,
    LLMMessage,
    LLMResult,
    LLMCallTimeoutException,
    LLMCallException,
    AnthropicLLMConfiguration,
    LLMessageTokenCounterBase,
    LLMConfigObject,
    LLMProviders,
)
from .anthropic import AnthropicAPIClientWrapper

from .anthropic_completion_llm import AnthropicCompletionLLM
from .anthropic_messages_llm import AnthropicMessagesLLM


class AnthropicTokenCounter(LLMessageTokenCounterBase):
    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        tokens = 0
        for msg in messages:
            tokens += self._client.count_tokens(msg.content)
        return tokens


class AnthropicLLM(LLMBase):
    def __init__(self, config: AnthropicLLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initialize a new instance.

        Args:
            config(AnthropicLLMConfiguration): configuration for the instance
        """
        super().__init__(name=name or f"{self.__class__.__name__}")
        self.config = config
        self._client = Anthropic(api_key=config.api_key.value, max_retries=0)
        self._api = self._get_api_wrapper()

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        try:
            response = self._api.post_chat_request(messages=messages)
            prompt_text = "\n".join([msg.content for msg in messages])
            return LLMResult(choices=response, consumptions=self.to_consumptions(prompt_text, response))
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self.config.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, prompt: str, responses: List[str]) -> Sequence[Consumption]:
        model = self.config.model.unwrap()
        prompt_tokens = self._client.count_tokens(prompt)
        completion_tokens = sum(self._client.count_tokens(r) for r in responses)
        return [
            Consumption(1, "call", f"{model}"),
            Consumption(prompt_tokens, "token", f"{model}:prompt_tokens"),
            Consumption(completion_tokens, "token", f"{model}:completion_tokens"),
            Consumption(prompt_tokens + completion_tokens, "token", f"{model}:total_tokens"),
        ]

    def _get_api_wrapper(self) -> AnthropicAPIClientWrapper:
        if self.config.model.value == "claude-2":
            return AnthropicCompletionLLM(client=self._client, config=self.config)
        return AnthropicMessagesLLM(client=self._client, config=self.config)

    @staticmethod
    def from_env() -> AnthropicLLM:
        """
        Helper function that create a new instance by getting the configuration from environment variables.

        Returns:
            AnthropicLLM
        """

        return AnthropicLLM(AnthropicLLMConfiguration.from_env())

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> AnthropicLLM:
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Anthropic):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Anthropic}")

        config = AnthropicLLMConfiguration.from_spec(config_object.spec)
        return AnthropicLLM(config=config, name=config_object.metadata.name)
