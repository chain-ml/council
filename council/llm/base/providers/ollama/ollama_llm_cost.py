from __future__ import annotations

from typing import Any, Final, List, Mapping, Optional

from council.contexts import Consumption

from ...llm_cost import LLMConsumptionCalculatorBase, LLMCostCard, TokenKind


class OllamaConsumptionCalculator(LLMConsumptionCalculatorBase):
    DURATION_KEYS: Final[List[str]] = ["prompt_eval_duration", "eval_duration", "load_duration", "total_duration"]

    def get_consumptions(self, duration: float, response: Mapping[str, Any]) -> List[Consumption]:
        """
        Get consumptions specific for ollama:
            - 1 call
            - specified duration
            - prompt, completion and total tokens if response contains "prompt_eval_count" and "eval_count" keys
            - ollama durations if response contains DURATION_KEYS.
        """

        return (
            self.get_default_consumptions(duration)
            + self.get_prompt_consumptions(response)
            + self.get_duration_consumptions(response)
        )

    def get_prompt_consumptions(self, response: Mapping[str, Any]) -> List[Consumption]:
        if not all(key in response for key in ["prompt_eval_count", "eval_count"]):
            return []

        prompt_tokens = response["prompt_eval_count"]
        completion_tokens = response["eval_count"]
        return [
            Consumption.token(prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_duration_consumptions(self, response: Mapping[str, Any]) -> List[Consumption]:
        if not all(key in response for key in self.DURATION_KEYS):
            return []

        # from nanoseconds to seconds
        return [Consumption.duration(response[key] / 1e9, f"{self.model}:ollama_{key}") for key in self.DURATION_KEYS]

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return None
