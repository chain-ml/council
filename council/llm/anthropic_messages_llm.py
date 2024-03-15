from __future__ import annotations

from typing import Sequence, List, Iterable, Literal

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import MessageParam

from council.llm import (
    LLMMessage,
    LLMMessageRole,
    AnthropicLLMConfiguration,
)
from council.llm.anthropic import AnthropicAPIClientWrapper


class AnthropicMessagesLLM(AnthropicAPIClientWrapper):
    """
    Implementation for an Anthropic LLM with messages API.
    Needs to used for models like `claude-2.1` or `claude-3-xyz` versions.

    Notes:
        More details: https://docs.anthropic.com/claude/docs
        and https://docs.anthropic.com/claude/reference/messages_post
    """

    def __init__(self, config: AnthropicLLMConfiguration, client: Anthropic) -> None:
        self._config = config
        self._client = client

    def post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]:
        messages_formatted = self._to_anthropic_messages(messages)
        completion = self._client.messages.create(
            messages=messages_formatted,
            model=self._config.model.unwrap(),
            max_tokens=self._config.max_tokens.unwrap(),
            timeout=self._config.timeout.value,
            temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
            top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
            top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
        )
        return [content.text for content in completion.content]

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> Iterable[MessageParam]:
        result: List[MessageParam] = []
        temp_content = ""
        role: Literal["user", "assistant"] = "user"

        for message in messages:
            if message.is_of_role(LLMMessageRole.System):
                temp_content += message.content
            else:
                temp_content += message.content
                result.append(MessageParam(role=role, content=temp_content))
                temp_content = ""
                role = "assistant" if role == "user" else "user"

        if temp_content:
            result.append(MessageParam(role=role, content=temp_content))

        return result
