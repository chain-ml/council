from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    LLMMessage,
    LLMMessageTokenCounterBase,
    LLMProviders,
    LLMResult,
)

from .anthropic import AnthropicAPIClientWrapper
from .anthropic_completion_llm import AnthropicCompletionLLM
from .anthropic_messages_llm import AnthropicMessagesLLM


class AnthropicTokenCounter(LLMMessageTokenCounterBase):
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
        "claude-3-haiku-20240307": LLMCostCard(input=0.25, output=1.25),
        "claude-3-sonnet-20240229": LLMCostCard(input=3.00, output=15.00),
        "claude-3-5-sonnet-20240620": LLMCostCard(input=3.00, output=15.00),
        "claude-3-5-sonnet-20241022": LLMCostCard(input=3.00, output=15.00),
        "claude-3-opus-20240229": LLMCostCard(input=15.00, output=75.00),
    }

    # input - cache write; output - cache read; note - not all model support prompt caching
    COSTS_CACHING: Mapping[str, LLMCostCard] = {
        "claude-3-haiku-20240307": LLMCostCard(input=0.30, output=0.03),
        "claude-3-5-sonnet-20240620": LLMCostCard(input=3.75, output=0.30),
        "claude-3-5-sonnet-20241022": LLMCostCard(input=3.75, output=0.30),
        "claude-3-opus-20240229": LLMCostCard(input=18.75, output=1.50),
    }

    def find_model_costs(self, model_name: str) -> Optional[LLMCostCard]:
        return self.COSTS.get(model_name)

    def get_caching_cost_consumptions(
        self,
        model: str,
        *,
        cache_creation_prompt_tokens: int,
        cache_read_prompt_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> List[Consumption]:
        """Get list of USD consumptions specific for Anthropic prompt caching."""
        cost_card = self.find_model_costs(model)
        caching_cost_card = self.COSTS_CACHING.get(model)

        if cost_card is None or caching_cost_card is None:
            return []

        prompt_tokens_cost, completion_tokens_cost = cost_card.get_costs(prompt_tokens, completion_tokens)
        cache_creation_prompt_tokens_cost, cache_read_prompt_tokens_cost = caching_cost_card.get_costs(
            cache_creation_prompt_tokens, cache_read_prompt_tokens
        )

        total_cost = (
            prompt_tokens_cost
            + completion_tokens_cost
            + cache_creation_prompt_tokens_cost
            + cache_read_prompt_tokens_cost
        )

        return [
            Consumption(cache_creation_prompt_tokens_cost, "USD", f"{model}:cache_creation_prompt_tokens_cost"),
            Consumption(cache_read_prompt_tokens_cost, "USD", f"{model}:cache_read_prompt_tokens_cost"),
            Consumption(prompt_tokens_cost, "USD", f"{model}:prompt_tokens_cost"),
            Consumption(completion_tokens_cost, "USD", f"{model}:completion_tokens_cost"),
            Consumption(total_cost, "USD", f"{model}:total_tokens_cost"),
        ]


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
            usage = response.raw_response["usage"] if response.raw_response is not None else {}
            return LLMResult(
                choices=response.choices,
                consumptions=self.to_consumptions(usage),
                raw_response=response.raw_response,
            )
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self._configuration.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, usage: Dict[str, int]) -> Sequence[Consumption]:
        if "input_tokens" not in usage or "output_tokens" not in usage:
            return []

        model = self._configuration.model_name()
        if "cache_creation_input_tokens" in usage:
            return self.to_cache_consumptions(
                model,
                cache_creation_prompt_tokens=usage["cache_creation_input_tokens"],
                cache_read_prompt_tokens=usage["cache_read_input_tokens"],
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
            )

        return self.to_default_consumptions(
            model, prompt_tokens=usage["input_tokens"], completion_tokens=usage["output_tokens"]
        )

    @staticmethod
    def to_default_consumptions(model: str, *, prompt_tokens: int, completion_tokens: int) -> Sequence[Consumption]:
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

    @staticmethod
    def to_cache_consumptions(
        model: str,
        *,
        cache_creation_prompt_tokens: int,
        cache_read_prompt_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Sequence[Consumption]:
        total = cache_creation_prompt_tokens + cache_read_prompt_tokens + prompt_tokens + completion_tokens
        base_consumptions = [
            Consumption(1, "call", f"{model}"),
            Consumption(cache_creation_prompt_tokens, "token", f"{model}:cache_creation_prompt_tokens"),
            Consumption(cache_read_prompt_tokens, "token", f"{model}:cache_read_prompt_tokens"),
            Consumption(prompt_tokens, "token", f"{model}:prompt_tokens"),
            Consumption(completion_tokens, "token", f"{model}:completion_tokens"),
            Consumption(total, "token", f"{model}:total_tokens"),
        ]

        consumptions = base_consumptions + AnthropicCostManager().get_caching_cost_consumptions(
            model,
            cache_creation_prompt_tokens=cache_creation_prompt_tokens,
            cache_read_prompt_tokens=cache_read_prompt_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # filter zero consumptions that could occur for cache tokens
        return list(filter(lambda consumption: consumption.value > 0, consumptions))

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
