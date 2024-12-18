from __future__ import annotations

import os
from typing import Final, Mapping, Optional

from ...llm_cost import DefaultLLMConsumptionCalculator, LLMCostCard, LLMCostManagerObject

GEMINI_COSTS_FILENAME: Final[str] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "gemini-costs.yaml"
)


class GeminiConsumptionCalculator(DefaultLLMConsumptionCalculator):
    _cost_manager = LLMCostManagerObject.from_yaml(GEMINI_COSTS_FILENAME)
    COSTS_UNDER_128k: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("under_128k")
    COSTS_OVER_128k: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("over_128k")

    def __init__(self, model: str, num_tokens: int) -> None:
        super().__init__(model)
        self.num_tokens = num_tokens

    def find_model_costs(self) -> Optional[LLMCostCard]:
        if self.num_tokens <= 128_000:
            return self.COSTS_UNDER_128k.get(self.model)
        return self.COSTS_OVER_128k.get(self.model)
