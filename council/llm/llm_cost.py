from __future__ import annotations

import abc
import os
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import yaml
from council.contexts import Consumption
from council.utils import DataObject, DataObjectSpecBase

DATA_PATH: Final[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANTHROPIC_COSTS_FILENAME: Final[str] = "anthropic-costs.yaml"
GEMINI_COSTS_FILENAME: Final[str] = "gemini-costs.yaml"
OPENAI_COSTS_FILENAME: Final[str] = "openai-costs.yaml"


class LLMCostCard:
    """LLM cost per million token"""

    def __init__(self, input: float, output: float) -> None:
        self._input = input
        self._output = output

    @property
    def input(self) -> float:
        """Cost per million input (prompt) tokens."""
        return self._input

    @property
    def output(self) -> float:
        """Cost per million output (completion) tokens."""
        return self._output

    def __str__(self) -> str:
        return f"${self.input}/${self.output} per 1m tokens"

    def input_cost(self, tokens: int) -> float:
        """Get prompt_tokens_cost for a given amount of input tokens."""
        return tokens * self.input / 1e6

    def output_cost(self, tokens: int) -> float:
        """Get completion_token_cost for a given amount of completion tokens."""
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

    @abc.abstractmethod
    def get_consumptions(self, *args, **kwargs) -> List[Consumption]:
        """Each calculator will implement with its own parameters."""
        pass

    @abc.abstractmethod
    def find_model_costs(self) -> Optional[LLMCostCard]:
        """Get LLMCostCard for self to calculate cost consumptions."""
        pass

    @staticmethod
    def filter_zeros(consumptions: List[Consumption]) -> List[Consumption]:
        return list(filter(lambda consumption: consumption.value > 0, consumptions))


class DefaultLLMConsumptionCalculator(LLMConsumptionCalculatorBase, abc.ABC):
    def get_consumptions(self, duration: float, *, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        """
        Get default consumptions:
            - 1 call
            - specified duration
            - prompt, completion and total tokens
            - corresponding costs if LLMCostCard can be found.
        """
        base_consumptions = self.get_base_consumptions(
            duration, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        cost_consumptions = self.get_cost_consumptions(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return base_consumptions + cost_consumptions

    def get_base_consumptions(
        self, duration: float, *, prompt_tokens: int, completion_tokens: int
    ) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.duration(duration, self.model),
            Consumption.token(prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_cost_consumptions(self, *, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        prompt_tokens_cost, completion_tokens_cost = cost_card.get_costs(prompt_tokens, completion_tokens)
        return [
            Consumption.cost(prompt_tokens_cost, self.format_kind(TokenKind.prompt, cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind(TokenKind.completion, cost=True)),
            Consumption.cost(prompt_tokens_cost + completion_tokens_cost, self.format_kind(TokenKind.total, cost=True)),
        ]


class LLMCostManagerSpec(DataObjectSpecBase):
    def __init__(self, costs: Dict[str, Dict[str, LLMCostCard]]) -> None:
        """
        Initializes a new instance of LLMCostManagerSpec

        Args:
            costs (Dict[str, Dict[str, LLMCostCard]]): collection of cost cards of shape
            {category: {model_1: LLMCostCard, model_2: LLMCostCard}, another_category: {...}}
        """
        self.costs = costs

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMCostManagerSpec:
        costs = {
            category: {
                model: LLMCostCard.from_dict(model_data) for model, model_data in category_data["models"].items()
            }
            for category, category_data in values.items()
        }

        return LLMCostManagerSpec(costs)

    def to_dict(self) -> Dict[str, Any]:
        return self.costs

    def __str__(self) -> str:
        return f"LLMCostCards for {len(self.costs.keys())} categories"


class LLMCostManagerObject(DataObject[LLMCostManagerSpec]):
    """
    Helper class to instantiate an LLMCostManagerObject from a YAML file
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMCostManagerObject:
        return super()._from_dict(LLMCostManagerSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMCostManagerObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMCostManager")
            return LLMCostManagerObject.from_dict(values)

    @staticmethod
    def anthropic():
        """Get LLMCostManager for Anthropic models"""
        return LLMCostManagerObject.from_yaml(os.path.join(DATA_PATH, ANTHROPIC_COSTS_FILENAME))

    @staticmethod
    def gemini():
        """Get LLMCostManager for Gemini models"""
        return LLMCostManagerObject.from_yaml(os.path.join(DATA_PATH, GEMINI_COSTS_FILENAME))

    @staticmethod
    def openai():
        """Get LLMCostManager for OpenAI models"""
        return LLMCostManagerObject.from_yaml(os.path.join(DATA_PATH, OPENAI_COSTS_FILENAME))

    def get_cost_map(self, category: str) -> Dict[str, LLMCostCard]:
        """Get cost mapping {model: LLMCostCard} for a given category"""
        if category not in self.spec.costs:
            raise ValueError(f"Unexpected category `{category}` for LLMCostManager")

        return self.spec.costs[category]
