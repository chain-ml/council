from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from council.contexts import Consumption, LLMContext
from council.llm import (
    LLMBase,
    LLMConfigObject,
    LLMConsumptionCalculatorBase,
    LLMCostCard,
    LLMMessage,
    LLMProviders,
    LLMResult,
    OllamaLLMConfiguration,
    TokenKind,
)
from council.utils.utils import DurationManager
from ollama import Client
from ollama._types import Message


class OllamaConsumptionCalculator(LLMConsumptionCalculatorBase):
    DURATION_KEYS = ["prompt_eval_duration", "eval_duration", "load_duration", "total_duration"]

    def get_ollama_consumptions(self, duration: float, response: Mapping[str, Any]) -> List[Consumption]:
        return (
            self.get_ollama_base_consumptions(duration)
            + self.get_ollama_prompt_consumptions(response)
            + self.get_ollama_duration_consumptions(response)
        )

    def get_ollama_base_consumptions(self, duration: float) -> List[Consumption]:
        return [Consumption.call(1, self.model), Consumption.duration(duration, self.model)]

    def get_ollama_prompt_consumptions(self, response: Mapping[str, Any]) -> List[Consumption]:
        if not all(key in response for key in ["prompt_eval_count", "eval_count"]):
            return []

        prompt_tokens = response["prompt_eval_count"]
        completion_tokens = response["eval_count"]
        return [
            Consumption.token(prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_ollama_duration_consumptions(self, response: Mapping[str, Any]) -> List[Consumption]:
        if not all(key in response for key in self.DURATION_KEYS):
            return []

        # from nanoseconds to seconds
        return [Consumption.duration(response[key] / 1e9, f"{self.model}:ollama_{key}") for key in self.DURATION_KEYS]

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return None


class OllamaLLM(LLMBase[OllamaLLMConfiguration]):
    def __init__(self, config: OllamaLLMConfiguration) -> None:
        """
        Initialize a new instance.

        Args:
            config (OllamaLLMConfiguration): configuration for the instance
        """
        super().__init__(name=f"{self.__class__.__name__}", configuration=config)

        self._client = Client()

    @property
    def client(self) -> Client:
        return self._client

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        messages_payload = self._build_messages_payload(messages)
        with DurationManager() as timer:
            response = self.client.chat(model=self.model_name, messages=messages_payload)

        return LLMResult(
            choices=[response["message"]["content"]], consumptions=self.to_consumptions(timer.duration, response)
        )

    @staticmethod
    def _build_messages_payload(messages: Sequence[LLMMessage]) -> List[Message]:
        return [Message(role=message.role.value, content=message.content) for message in messages]

    def to_consumptions(self, duration: float, response: Mapping[str, Any]) -> Sequence[Consumption]:
        model = self._configuration.model_name()

        calculator = OllamaConsumptionCalculator(model)
        return calculator.get_ollama_consumptions(duration, response)

    @staticmethod
    def from_env() -> OllamaLLM:
        """
        Helper function that create a new instance by getting the configuration from environment variables.

        Returns:
            OllamaLLM
        """

        return OllamaLLM(OllamaLLMConfiguration.from_env())

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> OllamaLLM:
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Ollama):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Ollama}")

        config = OllamaLLMConfiguration.from_spec(config_object.spec)
        return OllamaLLM(config=config)
