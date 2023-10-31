from typing import Any, Sequence

from anthropic import Anthropic
from council import LLMContext
from council.llm import LLMBase, LLMMessage, LLMMessageRole, LLMResult


_HUMAN_TURN = "\n\nHuman:"
_ASSISTANT_TURN = "\n\nAssistant:"


class AnthropicLLM(LLMBase):
    def __init__(self):
        super().__init__()
        self._client = Anthropic()

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        prompt = self._to_anthropic_messages(messages)
        completion = self._client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=300
        )

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
