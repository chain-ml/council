from __future__ import annotations

import os
from typing import Final, List, Mapping, Optional

from council.contexts import Consumption

from ...llm_cost import LLMConsumptionCalculatorBase, LLMCostCard, LLMCostManagerObject, TokenKind
from .anthropic import Usage

ANTHROPIC_COSTS_FILENAME: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "anthropic-costs.yaml"
)


class AnthropicConsumptionCalculator(LLMConsumptionCalculatorBase):
    _cost_manager = LLMCostManagerObject.from_yaml(ANTHROPIC_COSTS_FILENAME)
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
