from __future__ import annotations

import os
from typing import Any, Final, List, Mapping, Optional

from council.contexts import Consumption

from ...llm_cost import LLMConsumptionCalculatorBase, LLMCostCard, LLMCostManagerObject, TokenKind

OPENAI_COSTS_FILENAME: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "openai-costs.yaml"
)


class Usage:
    """
    Represents token usage statistics for an OpenAI API request, with normalized token counting.

    This class extends the standard OpenAI usage stats by separately tracking reasoning and cached tokens
    while maintaining consistent total token counts. Unlike the OpenAI implementation, this class:
    - Subtracts reasoning_tokens from completion_tokens to avoid double-counting
    - Subtracts cached_tokens from prompt_tokens to avoid double-counting
    """

    def __init__(
        self, completion_tokens: int, prompt_tokens: int, total_tokens: int, reasoning_tokens: int, cached_tokens: int
    ) -> None:
        self._completion = completion_tokens
        self._prompt = prompt_tokens
        self._total = total_tokens
        self._reasoning = reasoning_tokens
        self._cached = cached_tokens

    def __str__(self) -> str:
        return f'prompt_tokens="{self._prompt}" total_tokens="{self._total}" completion_tokens="{self._completion}"'

    @property
    def prompt_tokens(self) -> int:
        """Number of tokens in the prompt, excluding cached tokens."""
        return self._prompt

    @property
    def completion_tokens(self) -> int:
        """Number of tokens in the completion, excluding reasoning tokens."""
        return self._completion

    @property
    def total_tokens(self) -> int:
        """Total number of tokens used (cached + prompt + reasoning + completion)."""
        return self._total

    @property
    def reasoning_tokens(self) -> int:
        """Number of reasoning completion tokens."""
        return self._reasoning

    @property
    def cached_tokens(self) -> int:
        """Number of cached prompt tokens."""
        return self._cached

    @staticmethod
    def from_dict(obj: Any) -> Usage:
        _completion_tokens = int(obj.get("completion_tokens"))
        _prompt_tokens = int(obj.get("prompt_tokens"))
        _total_tokens = int(obj.get("total_tokens"))

        completion_tokens_details = obj.get("completion_tokens_details")
        _reasoning_tokens = completion_tokens_details["reasoning_tokens"] if completion_tokens_details else 0
        if _reasoning_tokens > 0:
            _completion_tokens -= _reasoning_tokens

        prompt_tokens_details = obj.get("prompt_tokens_details")
        _cached_tokens = prompt_tokens_details["cached_tokens"] if prompt_tokens_details else 0
        if _cached_tokens > 0:
            _prompt_tokens -= _cached_tokens

        return Usage(_completion_tokens, _prompt_tokens, _total_tokens, _reasoning_tokens, _cached_tokens)


class OpenAIConsumptionCalculator(LLMConsumptionCalculatorBase):
    _cost_manager = LLMCostManagerObject.from_yaml(OPENAI_COSTS_FILENAME)
    COSTS_gpt_35_turbo_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_35_turbo_family")
    COSTS_gpt_4_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_4_family")
    COSTS_gpt_4o_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_4o_family")
    COSTS_o1_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("o1_family")

    def find_model_costs(self) -> Optional[LLMCostCard]:
        if self.model.startswith("o1"):
            return self.COSTS_o1_FAMILY.get(self.model)
        elif self.model.startswith("gpt-4o"):
            return self.COSTS_gpt_4o_FAMILY.get(self.model)
        elif self.model.startswith("gpt-4"):
            return self.COSTS_gpt_4_FAMILY.get(self.model)
        elif self.model.startswith("gpt-3.5-turbo"):
            return self.COSTS_gpt_35_turbo_FAMILY.get(self.model)

        return None

    def get_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        """
        Get consumptions specific for OpenAI:
            - 1 call
            - specified duration
            - cache_read_prompt, prompt, reasoning, completion and total tokens
            - corresponding costs if LLMCostCard can be found
        """
        consumptions = self.get_base_consumptions(duration, usage) + self.get_cost_consumptions(usage)
        return self.filter_zeros(consumptions)  # could occur for cache/reasoning tokens

    def get_base_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.duration(duration, self.model),
            Consumption.token(usage.cached_tokens, self.format_kind(TokenKind.cache_read_prompt)),
            Consumption.token(usage.prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(usage.reasoning_tokens, self.format_kind(TokenKind.reasoning)),
            Consumption.token(usage.completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(usage.total_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_cost_consumptions(self, usage: Usage) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        cached_tokens_cost = cost_card.input_cost(usage.cached_tokens) / 2
        prompt_tokens_cost = cost_card.input_cost(usage.prompt_tokens)
        reasoning_tokens_cost = cost_card.output_cost(usage.reasoning_tokens)
        completion_tokens_cost = cost_card.output_cost(usage.completion_tokens)
        total_cost = sum([cached_tokens_cost, prompt_tokens_cost, reasoning_tokens_cost, completion_tokens_cost])

        return [
            Consumption.cost(cached_tokens_cost, self.format_kind(TokenKind.cache_read_prompt, cost=True)),
            Consumption.cost(prompt_tokens_cost, self.format_kind(TokenKind.prompt, cost=True)),
            Consumption.cost(reasoning_tokens_cost, self.format_kind(TokenKind.reasoning, cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind(TokenKind.completion, cost=True)),
            Consumption.cost(total_cost, self.format_kind(TokenKind.total, cost=True)),
        ]
