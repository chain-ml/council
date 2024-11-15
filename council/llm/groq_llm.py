from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from council.contexts import Consumption, LLMContext
from council.llm import (
    DefaultLLMConsumptionCalculator,
    GroqLLMConfiguration,
    LLMBase,
    LLMConfigObject,
    LLMCostCard,
    LLMCostManagerObject,
    LLMMessage,
    LLMMessageRole,
    LLMProviders,
    LLMResult,
)
from council.utils.utils import DurationManager
from groq import Groq
from groq.types import CompletionUsage
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.chat_completion import ChatCompletion, Choice


class GroqConsumptionCalculator(DefaultLLMConsumptionCalculator):
    _cost_manager = LLMCostManagerObject.groq()
    COSTS: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("default")

    def __init__(self, model: str) -> None:
        super().__init__(model)

    # TODO: naming
    def get_groq_consumptions(self, duration: float, usage: Optional[CompletionUsage]) -> List[Consumption]:
        if usage is None:
            return [Consumption.call(1, self.model), Consumption.duration(duration, self.model)]

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        return (
            self.get_base_consumptions(duration, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            + self.get_duration_consumptions(usage)
            + self.get_cost_consumptions(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        )

    def get_duration_consumptions(self, usage: CompletionUsage) -> List[Consumption]:
        """Optional duration consumptions specific to Groq."""
        consumptions = []
        if usage.queue_time is not None:
            consumptions.append(Consumption.duration(usage.queue_time, f"{self.model}:groq_queue_time"))
        if usage.prompt_time is not None:
            consumptions.append(Consumption.duration(usage.prompt_time, f"{self.model}:groq_prompt_time"))
        if usage.completion_time is not None:
            consumptions.append(Consumption.duration(usage.completion_time, f"{self.model}:groq_completion_time"))
        if usage.total_time is not None:
            consumptions.append(Consumption.duration(usage.total_time, f"{self.model}:groq_total_time"))
        return consumptions

    def find_model_costs(self) -> Optional[LLMCostCard]:
        return self.COSTS.get(self.model)


class GroqLLM(LLMBase[GroqLLMConfiguration]):
    def __init__(self, config: GroqLLMConfiguration) -> None:
        """
        Initialize a new instance.

        Args:
            config(GroqLLMConfiguration): configuration for the instance
        """
        super().__init__(name=f"{self.__class__.__name__}", configuration=config)
        self._client = Groq(api_key=config.api_key.value)

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        formatted_messages = self._build_messages_payload(messages)

        with DurationManager() as timer:
            response = self._client.chat.completions.create(
                messages=formatted_messages,
                model=self._configuration.model_name(),
                temperature=self._configuration.temperature.value,
            )

        return LLMResult(
            choices=self._to_choices(response.choices), consumptions=self._to_consumptions(timer.duration, response)
        )

    @staticmethod
    def _build_messages_payload(messages: Sequence[LLMMessage]) -> List[ChatCompletionMessageParam]:
        def _llm_message_to_groq_message(message: LLMMessage) -> ChatCompletionMessageParam:
            if message.is_of_role(LLMMessageRole.System):
                return ChatCompletionSystemMessageParam(role="system", content=message.content)
            elif message.is_of_role(LLMMessageRole.User):
                return ChatCompletionUserMessageParam(role="user", content=message.content)
            elif message.is_of_role(LLMMessageRole.Assistant):
                return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)

            raise ValueError(f"Unknown LLMessage role: `{message.role.value}`")

        return [_llm_message_to_groq_message(message) for message in messages]

    @staticmethod
    def _to_choices(choices: List[Choice]) -> List[str]:
        return [choice.message.content if choice.message.content is not None else "" for choice in choices]

    @staticmethod
    def _to_consumptions(duration: float, response: ChatCompletion) -> Sequence[Consumption]:
        calculator = GroqConsumptionCalculator(response.model)
        return calculator.get_groq_consumptions(duration, response.usage)

    @staticmethod
    def from_env() -> GroqLLM:
        """
        Helper function that create a new instance by getting the configuration from environment variables.

        Returns:
            GroqLLM
        """
        return GroqLLM(GroqLLMConfiguration.from_env())

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> GroqLLM:
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Groq):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Groq}")

        config = GroqLLMConfiguration.from_spec(config_object.spec)
        return GroqLLM(config=config)
