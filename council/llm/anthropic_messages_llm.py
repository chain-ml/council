from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Sequence

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import MessageParam, TextBlock
from council.llm import AnthropicLLMConfiguration, LLMMessage, LLMMessageRole
from council.llm.anthropic import AnthropicAPIClientResult, AnthropicAPIClientWrapper
from council.llm.llm_message import LLMCacheControlData


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

    def post_chat_request(self, messages: Sequence[LLMMessage]) -> AnthropicAPIClientResult:
        system_params = self._to_anthropic_system_messages(messages)
        messages_formatted = self._to_anthropic_messages(messages)

        endpoint = (
            self._client.messages if not self._use_caching(messages) else self._client.beta.prompt_caching.messages
        )

        completion = endpoint.create(  # type: ignore
            **system_params,
            messages=messages_formatted,
            model=self._config.model.unwrap(),
            max_tokens=self._config.max_tokens.unwrap(),
            timeout=self._config.timeout.value,
            temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
            top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
            top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
        )
        choices = [content.text for content in completion.content if isinstance(content, TextBlock)]

        return AnthropicAPIClientResult(choices=choices, raw_response=completion.to_dict())

    @staticmethod
    def _to_anthropic_system_messages(messages: Sequence[LLMMessage]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns a dict suitable for unpacking as keyword arguments in the Anthropic client's create method:
            - Dict with a "system" key mapping to a list of formatted system messages ({"system": content})
            - Empty dict if there's no system messages to format
        """

        result: List[Dict[str, Any]] = []
        for message in messages:
            if not message.is_of_role(LLMMessageRole.System):
                continue

            content_item: Dict[str, Any] = {"type": "text", "text": message.content}
            for data in message.data:
                if isinstance(data, LLMCacheControlData):
                    content_item["cache_control"] = data.cache_control

            result.append(content_item)

        return {"system": result} if len(result) > 0 else {}

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> Iterable[MessageParam]:
        result: List[MessageParam] = []

        for message in messages:
            if message.is_of_role(LLMMessageRole.System):
                continue

            role: Literal["user", "assistant"] = "user" if message.is_of_role(LLMMessageRole.User) else "assistant"

            additional_content: List[Dict[str, Any]] = []
            text_content_item: Dict[str, Any] = {"type": "text", "text": message.content}

            for data in message.data:
                if data.is_image:
                    image_item = {
                        "type": "image",
                        "source": {"type": "base64", "media_type": data.mime_type, "data": data.content},
                    }
                    additional_content.append(image_item)
                elif isinstance(data, LLMCacheControlData):
                    text_content_item["cache_control"] = data.cache_control

            content: Iterable[Any] = [text_content_item] + additional_content
            result.append(MessageParam(role=role, content=content))

        return result

    @staticmethod
    def _use_caching(messages: Sequence[LLMMessage]) -> bool:
        for message in messages:
            for data in message.data:
                if isinstance(data, LLMCacheControlData):
                    return True
        return False
