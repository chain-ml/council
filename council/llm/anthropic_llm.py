from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from anthropic import Anthropic, APIStatusError, APITimeoutError
from council.contexts import Consumption, LLMContext
from council.llm import (
    AnthropicLLMConfiguration,
    LLMBase,
    LLMCallException,
    LLMCallTimeoutException,
    LLMConfigObject,
    LLMCostCard,
    LLMCostManager,
    LLMessageTokenCounterBase,
    LLMMessage,
    LLMProviders,
    LLMResult,
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


class AnthropicCostManager(LLMCostManager):
    # https://www.anthropic.com/pricing#anthropic-api
    COSTS: Mapping[str, LLMCostCard] = {
        # haiku
        "claude-3-haiku-20240307": LLMCostCard(input=0.25, output=1.25),
        # sonnet
        "claude-3-sonnet-20240229": LLMCostCard(input=3.00, output=15.00),
        "claude-3-5-sonnet-20240620": LLMCostCard(input=3.00, output=15.00),
        "claude-3-5-sonnet-20241022": LLMCostCard(input=3.00, output=15.00),
        # opus
        "claude-3-opus-20240229": LLMCostCard(input=15.00, output=75.00),
    }

    def find_model_costs(self, model_name: str) -> Optional[LLMCostCard]:
        model_name = model_name.split(" with fallback")[0]
        return self.COSTS.get(model_name)


class AnthropicLLM(LLMBase[AnthropicLLMConfiguration]):
    def __init__(self, config: AnthropicLLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initialize a new instance.

        Args:
            config(AnthropicLLMConfiguration): configuration for the instance
        """
        super().__init__(name=name or f"{self.__class__.__name__}", configuration=config)
        self._client = Anthropic(api_key=config.api_key.value, max_retries=0)
        self._api = self._get_api_wrapper()

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        try:
            response = self._api.post_chat_request(messages=messages)
            prompt_text = "\n".join([msg.content for msg in messages])
            return LLMResult(
                choices=response.choices,
                consumptions=self.to_consumptions(prompt_text, response.choices),
                raw_response=response.raw_response,
            )
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self._configuration.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, prompt: str, responses: List[str]) -> Sequence[Consumption]:
        model = self._configuration.model_name()
        prompt_tokens = self._client.count_tokens(prompt)
        completion_tokens = sum(self._client.count_tokens(r) for r in responses)
        base_consumptions = [
            Consumption(1, "call", f"{model}"),
            Consumption(prompt_tokens, "token", f"{model}:prompt_tokens"),
            Consumption(completion_tokens, "token", f"{model}:completion_tokens"),
            Consumption(prompt_tokens + completion_tokens, "token", f"{model}:total_tokens"),
        ]

        cost_card = AnthropicCostManager().find_model_costs(model)
        if cost_card is None:
            return base_consumptions

        return base_consumptions + cost_card.get_consumptions(model, prompt_tokens, completion_tokens)

    def _get_api_wrapper(self) -> AnthropicAPIClientWrapper:
        if self._configuration is not None and self._configuration.model_name() == "claude-2":
            return AnthropicCompletionLLM(client=self._client, config=self.configuration)
        return AnthropicMessagesLLM(client=self._client, config=self.configuration)

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
