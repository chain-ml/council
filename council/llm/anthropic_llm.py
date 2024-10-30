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
    LLMConsumptionCalculatorBase,
    LLMCostCard,
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


class AnthropicConsumptionCalculator(LLMConsumptionCalculatorBase):
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

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS.get(self.model)

    def find_caching_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS_CACHING.get(self.model)

    def get_cache_consumptions(self, usage: Dict[str, int]) -> List[Consumption]:
        """
        Get consumptions specific for Anthropic prompt caching:
            - 1 call
            - cache_creation_prompt, cache_read_prompt, prompt, completion and total tokens
            - costs if both regular and caching LLMCostCards can be found
        """
        consumptions = self.get_cache_token_consumptions(usage) + self.get_cache_cost_consumptions(usage)
        return self.filter_zeros(consumptions)  # could occur for cache tokens

    def get_cache_token_consumptions(self, usage: Dict[str, int]) -> List[Consumption]:
        total = sum(
            [
                usage["cache_creation_prompt_tokens"],
                usage["cache_read_prompt_tokens"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
            ]
        )
        return [
            Consumption.call(1, self.model),
            Consumption.token(usage["cache_creation_prompt_tokens"], self.format_kind("cache_creation_prompt")),
            Consumption.token(usage["cache_read_prompt_tokens"], self.format_kind("cache_read_prompt")),
            Consumption.token(usage["prompt_tokens"], self.format_kind("prompt")),
            Consumption.token(usage["completion_tokens"], self.format_kind("completion")),
            Consumption.token(total, self.format_kind("total")),
        ]

    def get_cache_cost_consumptions(self, usage: Dict[str, int]) -> List[Consumption]:
        cost_card = self.find_model_costs()
        caching_cost_card = self.find_caching_costs()

        if cost_card is None or caching_cost_card is None:
            return []

        prompt_tokens_cost = cost_card.input_cost(usage["prompt_tokens"])
        completion_tokens_cost = cost_card.output_cost(usage["completion_tokens"])
        cache_creation_prompt_tokens_cost = caching_cost_card.input_cost(usage["cache_creation_prompt_tokens"])
        cache_read_prompt_tokens_cost = caching_cost_card.output_cost(usage["cache_read_prompt_tokens"])

        total_cost = sum(
            [
                prompt_tokens_cost,
                completion_tokens_cost,
                cache_creation_prompt_tokens_cost,
                cache_read_prompt_tokens_cost,
            ]
        )

        return [
            Consumption.cost(cache_creation_prompt_tokens_cost, self.format_kind("cache_creation_prompt", cost=True)),
            Consumption.cost(cache_read_prompt_tokens_cost, self.format_kind("cache_read_prompt", cost=True)),
            Consumption.cost(prompt_tokens_cost, self.format_kind("prompt", cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind("completion", cost=True)),
            Consumption.cost(total_cost, self.format_kind("total", cost=True)),
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
        consumption_calculator = AnthropicConsumptionCalculator(model)
        if "cache_creation_input_tokens" in usage:
            return consumption_calculator.get_cache_consumptions(usage)

        return consumption_calculator.get_consumptions(usage["input_tokens"], usage["output_tokens"])

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
