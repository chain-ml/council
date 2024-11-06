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
    LLMConsumptionCalculatorBase,
    LLMCostCard,
    LLMCostManagerObject,
    LLMMessage,
    LLMMessageTokenCounterBase,
    LLMProviders,
    LLMResult,
    TokenKind,
)
from council.utils.utils import DurationManager

from .anthropic import AnthropicAPIClientWrapper, Usage
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


class AnthropicConsumptionCalculator(LLMConsumptionCalculatorBase):
    _cost_manager = LLMCostManagerObject.anthropic()
    COSTS: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("default")
    COSTS_CACHING: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("caching")

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS.get(self.model)

    def find_caching_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS_CACHING.get(self.model)

    def get_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        """
        Get consumptions specific for Anthropic supporting prompt caching:
            - 1 call
            - specified duration
            - cache_creation_prompt, cache_read_prompt, prompt, completion and total tokens
            - corresponding costs if both regular and caching LLMCostCards can be found
        """
        consumptions = self.get_base_consumptions(duration, usage) + self.get_cost_consumptions(usage)
        return self.filter_zeros(consumptions)  # could occur for cache tokens

    def get_base_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.duration(duration, self.model),
            Consumption.token(usage.cache_creation_prompt_tokens, self.format_kind(TokenKind.cache_creation_prompt)),
            Consumption.token(usage.cache_read_prompt_tokens, self.format_kind(TokenKind.cache_read_prompt)),
            Consumption.token(usage.prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(usage.completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(usage.total_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_cost_consumptions(self, usage: Usage) -> List[Consumption]:
        cost_card = self.find_model_costs()
        caching_cost_card = self.find_caching_costs()

        if cost_card is None or caching_cost_card is None:
            return []

        prompt_tokens_cost = cost_card.input_cost(usage.prompt_tokens)
        completion_tokens_cost = cost_card.output_cost(usage.completion_tokens)
        cache_creation_prompt_tokens_cost = caching_cost_card.input_cost(usage.cache_creation_prompt_tokens)
        cache_read_prompt_tokens_cost = caching_cost_card.output_cost(usage.cache_read_prompt_tokens)

        total_cost = sum(
            [
                prompt_tokens_cost,
                completion_tokens_cost,
                cache_creation_prompt_tokens_cost,
                cache_read_prompt_tokens_cost,
            ]
        )

        return [
            Consumption.cost(
                cache_creation_prompt_tokens_cost, self.format_kind(TokenKind.cache_creation_prompt, cost=True)
            ),
            Consumption.cost(cache_read_prompt_tokens_cost, self.format_kind(TokenKind.cache_read_prompt, cost=True)),
            Consumption.cost(prompt_tokens_cost, self.format_kind(TokenKind.prompt, cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind(TokenKind.completion, cost=True)),
            Consumption.cost(total_cost, self.format_kind(TokenKind.total, cost=True)),
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
            with DurationManager() as timer:
                response = self._api.post_chat_request(messages=messages)
            return LLMResult(
                choices=response.choices,
                consumptions=self.to_consumptions(timer.duration, response.usage),
                raw_response=response.raw_response,
            )
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self._configuration.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, duration: float, usage: Usage) -> Sequence[Consumption]:
        model = self._configuration.model_name()
        consumption_calculator = AnthropicConsumptionCalculator(model)
        return consumption_calculator.get_consumptions(duration, usage)

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
