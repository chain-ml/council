from __future__ import annotations

import abc
from enum import Enum
from typing import Dict, List, Optional, Tuple

from council.contexts import Consumption


class LLMCostCard:
    """LLM cost per million token"""

    def __init__(self, input: float, output: float) -> None:
        self._input = input
        self._output = output

    @property
    def input(self) -> float:
        return self._input

    @property
    def output(self) -> float:
        return self._output

    def __str__(self) -> str:
        return f"${self.input}/${self.output} per 1m tokens"

    def input_cost(self, tokens: int) -> float:
        return tokens * self.input / 1e6

    def output_cost(self, tokens: int) -> float:
        return tokens * self.output / 1e6

    def get_costs(self, prompt_tokens: int, completion_tokens: int) -> Tuple[float, float]:
        """Return tuple of (prompt_tokens_cost, completion_token_cost)"""
        return self.input_cost(prompt_tokens), self.output_cost(completion_tokens)

    @staticmethod
    def from_dict(data: Dict[str, float]) -> LLMCostCard:
        return LLMCostCard(input=data["input"], output=data["output"])


class TokenKind(str, Enum):
    prompt = "prompt"
    """Prompt tokens"""

    completion = "completion"
    """Completion tokens"""

    total = "total"
    """Total tokens"""

    reasoning = "reasoning"
    """Reasoning tokens, specific for OpenAI o1 models"""

    cache_creation_prompt = "cache_creation_prompt"
    """Cache creation prompt tokens, specific for Anthropic prompt caching"""

    cache_read_prompt = "cache_read_prompt"
    """Cache read prompt tokens, specific for Anthropic and OpenAI prompt caching"""


class LLMConsumptionCalculatorBase(abc.ABC):
    """Helper class to manage LLM consumptions."""

    def __init__(self, model: str):
        self.model = model

    def format_kind(self, token_kind: TokenKind, cost: bool = False) -> str:
        """Format Consumption.kind - from 'prompt' to '{self.model}:prompt_tokens'"""
        kind = token_kind.value
        return f"{self.model}:{kind}_tokens" if not cost else f"{self.model}:{kind}_tokens_cost"

    def get_consumptions(self, duration: float, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        """
        Get default consumptions:
            - 1 call
            - specified duration
            - prompt, completion and total tokens
            - cost for prompt, completion and total tokens if LLMCostCard can be found
        """
        return self.get_base_consumptions(duration, prompt_tokens, completion_tokens) + self.get_cost_consumptions(
            prompt_tokens, completion_tokens
        )

    def get_base_consumptions(self, duration: float, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.duration(duration, self.model),
            Consumption.token(prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_cost_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        prompt_tokens_cost, completion_tokens_cost = cost_card.get_costs(prompt_tokens, completion_tokens)
        return [
            Consumption.cost(prompt_tokens_cost, self.format_kind(TokenKind.prompt, cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind(TokenKind.completion, cost=True)),
            Consumption.cost(prompt_tokens_cost + completion_tokens_cost, self.format_kind(TokenKind.total, cost=True)),
        ]

    @abc.abstractmethod
    def find_model_costs(self) -> Optional[LLMCostCard]:
        """Get LLMCostCard for self to calculate cost consumptions."""
        pass

    @staticmethod
    def filter_zeros(consumptions: List[Consumption]) -> List[Consumption]:
        return list(filter(lambda consumption: consumption.value > 0, consumptions))
