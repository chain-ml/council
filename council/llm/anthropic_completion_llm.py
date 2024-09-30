from typing import Sequence

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN
from council.llm import AnthropicLLMConfiguration, LLMMessage, LLMMessageRole
from council.llm.anthropic import AnthropicAPIClientResult, AnthropicAPIClientWrapper

_HUMAN_TURN = Anthropic.HUMAN_PROMPT
_ASSISTANT_TURN = Anthropic.AI_PROMPT


class AnthropicCompletionLLM(AnthropicAPIClientWrapper):
    """
    Implementation for an Anthropic LLM with LEGACY completion API.
    Needs to used for models like `claude-2` version.

    Notes:
        More details: https://docs.anthropic.com/claude/docs
        and https://docs.anthropic.com/claude/reference/complete_post
    """

    def __init__(self, config: AnthropicLLMConfiguration, client: Anthropic) -> None:
        self._config = config
        self._client = client

    def post_chat_request(self, messages: Sequence[LLMMessage]) -> AnthropicAPIClientResult:
        prompt = self._to_anthropic_messages(messages)
        result = self._client.completions.create(
            prompt=prompt,
            model=self._config.model.unwrap(),
            max_tokens_to_sample=self._config.max_tokens.unwrap(),
            timeout=self._config.timeout.value,
            temperature=self._config.temperature.unwrap_or(NOT_GIVEN),
            top_k=self._config.top_k.unwrap_or(NOT_GIVEN),
            top_p=self._config.top_p.unwrap_or(NOT_GIVEN),
        )
        return AnthropicAPIClientResult.from_completion(result)

    @staticmethod
    def _to_anthropic_messages(messages: Sequence[LLMMessage]) -> str:
        messages_count = len(messages)
        if messages_count == 0:
            raise RuntimeError("No message to process.")

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
