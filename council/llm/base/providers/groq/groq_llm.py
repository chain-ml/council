from __future__ import annotations

from typing import Any, List, Sequence

from council.contexts import Consumption, LLMContext
from council.utils.utils import DurationManager
from groq import Groq
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.chat_completion import ChatCompletion, Choice

from ...llm_base import LLMBase, LLMResult
from ...llm_message import LLMMessage, LLMMessageRole
from .groq_llm_configuration import GroqLLMConfiguration
from .groq_llm_cost import GroqConsumptionCalculator


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
                **self._configuration.params_to_args(),
                **kwargs,
            )

        return LLMResult(
            choices=self._to_choices(response.choices),
            consumptions=self._to_consumptions(timer.duration, response),
            raw_response=response.to_dict(),
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
        return calculator.get_consumptions(duration, response.usage)
