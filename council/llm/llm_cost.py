import abc
from typing import List, Optional, Tuple

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


class LLMConsumptionCalculatorBase(abc.ABC):
    """Helper class to manage LLM consumptions."""

    def __init__(self, model: str):
        self.model = model

    def format_kind(self, token_kind: str, cost: bool = False) -> str:
        """Format Consumption.kind - from 'prompt' to '{self.model}:prompt_tokens'"""
        options = [
            "prompt",
            "completion",
            "total",
            "reasoning",  # OpenAI o1
            "cache_creation_prompt",  # Anthropic prompt caching
            "cache_read_prompt",  # Anthropic & OpenAI prompt caching
        ]
        result = f"{self.model}:"
        if token_kind not in options:
            raise ValueError(
                f"Unknown kind `{token_kind}` for LLMConsumptionCalculator; expected one of `{','.join(options)}`"
            )

        result += f"{token_kind}_tokens"

        if cost:
            result += "_cost"

        return result

    def get_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        """
        Get default consumptions:
            - 1 call
            - prompt, completion and total tokens
            - cost for prompt, completion and total tokens if LLMCostCard can be found
        """
        return self.get_token_consumptions(prompt_tokens, completion_tokens) + self.get_cost_consumptions(
            prompt_tokens, completion_tokens
        )

    def get_token_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.token(prompt_tokens, self.format_kind("prompt")),
            Consumption.token(completion_tokens, self.format_kind("completion")),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind("total")),
        ]

    def get_cost_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        prompt_tokens_cost, completion_tokens_cost = cost_card.get_costs(prompt_tokens, completion_tokens)
        return [
            Consumption.cost(prompt_tokens_cost, self.format_kind("prompt", cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind("completion", cost=True)),
            Consumption.cost(prompt_tokens_cost + completion_tokens_cost, self.format_kind("total", cost=True)),
        ]

    @abc.abstractmethod
    def find_model_costs(self) -> Optional[LLMCostCard]:
        """Get LLMCostCard for self to calculate cost consumptions."""
        pass

    @staticmethod
    def filter_zeros(consumptions: List[Consumption]) -> List[Consumption]:
        return list(filter(lambda consumption: consumption.value > 0, consumptions))
