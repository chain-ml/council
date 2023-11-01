from __future__ import annotations

from typing import Any, Sequence

from anthropic import Anthropic, APITimeoutError, APIStatusError
from anthropic._types import NOT_GIVEN

from council.contexts import LLMContext, Consumption
from council.llm import (
    LLMBase,
    LLMMessage,
    LLMMessageRole,
    LLMResult,
    LLMCallTimeoutException,
    LLMCallException,
    AnthropicLLMConfiguration,
    LLMException,
    LLMessageTokenCounterBase,
)

_HUMAN_TURN = Anthropic.HUMAN_PROMPT
_ASSISTANT_TURN = Anthropic.AI_PROMPT


class AnthropicTokenCounter(LLMessageTokenCounterBase):
    def __init__(self, client: Anthropic):
        self._client = client

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        tokens = 0
        for msg in messages:
            tokens += self._client.count_tokens(msg.content)
        return tokens


class AnthropicLLM(LLMBase):
    """
    Implementation for an Anthropic LLM.

    Notes:
        More details: https://docs.anthropic.com/claude/docs
        and https://docs.anthropic.com/claude/reference/complete_post
    """

    def __init__(self, config: AnthropicLLMConfiguration):
        """
        Initialize a new instance.

        Args:
            config(AnthropicLLMConfiguration): configuration for the instance
        """
        super().__init__()
        self._config = config
        self._client = Anthropic(api_key=config.api_key.value, max_retries=0)

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        prompt = self._to_anthropic_messages(messages)
        try:
            completion = self._client.completions.create(
                prompt=prompt,
                model=self._config.model.unwrap(),
                max_tokens_to_sample=self._config.max_tokens.unwrap(),
                timeout=self._config.timeout.value,
                temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
                top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
                top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
            )
            response = completion.completion
            return LLMResult(choices=[response], consumptions=self.to_consumptions(prompt, response))
        except APITimeoutError as e:
            raise LLMCallTimeoutException(self._config.timeout.value) from e
        except APIStatusError as e:
            raise LLMCallException(code=e.status_code, error=e.message) from e

    def to_consumptions(self, prompt: str, response: str) -> Sequence[Consumption]:
        model = self._config.model.unwrap()
        prompt_tokens = self._client.count_tokens(prompt)
        completion_tokens = self._client.count_tokens(response)
        return [
            Consumption(1, "call", f"{model}"),
            Consumption(prompt_tokens, "token", f"{model}:prompt_tokens"),
            Consumption(completion_tokens, "token", f"{model}:completion_tokens"),
            Consumption(prompt_tokens + completion_tokens, "token", f"{model}:total_tokens"),
        ]

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> str:
        messages_count = len(messages)
        if messages_count == 0:
            raise LLMException("No message to process.")

        result = []
        if messages[0].is_of_role(LLMMessageRole.System) and messages_count > 1:
            result.append(f"{_HUMAN_TURN} {messages[0].content}\n{messages[1].content}")
            remaining = messages[2:]
        else:
            result.append(f"{_HUMAN_TURN} {messages[0].content}")
            remaining = messages[1:]

        for item in remaining:
            prefix = _HUMAN_TURN if item.is_of_role(LLMMessageRole.User) else _ASSISTANT_TURN
            result.append(f"{prefix} {item.content}")
        result.append(_ASSISTANT_TURN)

        return "".join(result)

    @staticmethod
    def from_env() -> AnthropicLLM:
        """
        Helper function that create a new instance by getting the configuration from environment variables.

        Returns:
            AnthropicLLM
        """

        return AnthropicLLM(AnthropicLLMConfiguration.from_env())
