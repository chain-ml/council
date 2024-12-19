from __future__ import annotations

from typing import Any, Optional, Sequence

from anthropic import Anthropic, APIStatusError, APITimeoutError
from council.contexts import Consumption, LLMContext
from council.utils.utils import DurationManager

from ...llm_base import LLMBase, LLMResult
from ...llm_exception import LLMCallException, LLMCallTimeoutException
from ...llm_message import LLMMessage
from .anthropic import AnthropicAPIClientWrapper, Usage
from .anthropic_completion_llm import AnthropicCompletionLLM
from .anthropic_llm_configuration import AnthropicLLMConfiguration
from .anthropic_llm_cost import AnthropicConsumptionCalculator
from .anthropic_messages_llm import AnthropicMessagesLLM


class AnthropicLLM(LLMBase[AnthropicLLMConfiguration]):
    def __init__(self, config: AnthropicLLMConfiguration, name: Optional[str] = None) -> None:
        """
        Initialize a new instance.

        Args:
            config(AnthropicLLMConfiguration): configuration for the instance
        """
        super().__init__(name=name or f"{self.__class__.__name__}", configuration=config)
        self._client = Anthropic(api_key=config.api_key.value, max_retries=0)
        self._api = self._get_api_wrapper()

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        try:
            with DurationManager() as timer:
                response = self._api.post_chat_request(messages=messages)
            return LLMResult(
                choices=response.choices,
                consumptions=self.to_consumptions(timer.duration, response.usage),
                raw_response=response.raw_response,
            )
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self._configuration.timeout.value, self._name) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message, llm_name=self._name) from e

    def to_consumptions(self, duration: float, usage: Usage) -> Sequence[Consumption]:
        model = self._configuration.model_name()
        consumption_calculator = AnthropicConsumptionCalculator(model)
        return consumption_calculator.get_consumptions(duration, usage)

    def _get_api_wrapper(self) -> AnthropicAPIClientWrapper:
        if self._configuration is not None and self._configuration.model_name() == "claude-2":
            return AnthropicCompletionLLM(client=self._client, config=self.configuration)
        return AnthropicMessagesLLM(client=self._client, config=self.configuration)
