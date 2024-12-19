from __future__ import annotations

import os
from typing import Dict, Final, List, Mapping, Optional

from council.contexts import Consumption
from groq.types import CompletionUsage

from ...llm_cost import DefaultLLMConsumptionCalculatorHelper, LLMCostCard, LLMCostManagerObject

GROQ_COSTS_FILENAME: Final[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "groq-costs.yaml")


class GroqConsumptionCalculator(DefaultLLMConsumptionCalculatorHelper):
    _cost_manager = LLMCostManagerObject.from_yaml(GROQ_COSTS_FILENAME)

    COSTS: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("default")

    def __init__(self, model: str) -> None:
        super().__init__(model)

    def get_consumptions(self, duration: float, usage: Optional[CompletionUsage]) -> List[Consumption]:
        if usage is None:
            return self.get_default_consumptions(duration)

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        return (
            self.get_base_consumptions(duration, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            + self.get_duration_consumptions(usage)
            + self.get_cost_consumptions(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        )

    def get_duration_consumptions(self, usage: CompletionUsage) -> List[Consumption]:
        """Optional duration consumptions specific to Groq."""
        usage_times: Dict[str, Optional[float]] = {
            "queue_time": usage.queue_time,
            "prompt_time": usage.prompt_time,
            "completion_time": usage.completion_time,
            "total_time": usage.total_time,
        }

        consumptions = []
        for key, value in usage_times.items():
            if value is not None:
                consumptions.append(Consumption.duration(value, f"{self.model}:groq_{key}"))

        return consumptions

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS.get(self.model)
