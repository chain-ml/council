from __future__ import annotations

from typing import Any, Sequence

from anthropic import Anthropic

from council.contexts import LLMContext
from council.llm import LLMBase, LLMMessage, LLMMessageRole, LLMResult
from .anthropic_llm_configuration import AnthropicLLMConfiguration

_HUMAN_TURN = "\n\nHuman:"
_ASSISTANT_TURN = "\n\nAssistant:"


class AnthropicLLM(LLMBase):
    """
    Implementation for an Anthropic LLM.

    Notes:
        More details: https://docs.anthropic.com/claude/docs
    """

    def __init__(self, config: AnthropicLLMConfiguration):
        """
        Initialize a new instance.

        Args:
            config(AnthropicLLMConfiguration): configuration for the instance
        """
        super().__init__()
        self._config = config
        self._client = Anthropic(api_key=self._config.api_key, timeout=config.timeout)

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        prompt = self._to_anthropic_messages(messages)
        completion = self._client.completions.create(model=self._config.model, prompt=prompt, max_tokens_to_sample=300)

        return LLMResult(choices=[completion.completion])

    @staticmethod
    def _to_anthropic_messages(messages) -> str:
        result = []
        if messages[0].is_of_role(LLMMessageRole.System):
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
