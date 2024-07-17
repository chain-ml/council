from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol, Sequence

from council import LLMContext
from council.llm import (
    LLMBase,
    LLMConfigurationBase,
    LLMessageTokenCounterBase,
    LLMException,
    LLMMessage,
    LLMResult,
    LLMTokenLimitException,
)


class LLMMessagesToStr(Protocol):
    def __call__(self, messages: Sequence[LLMMessage]) -> Sequence[str]: ...


class MockTokenCounter(LLMessageTokenCounterBase):
    def __init__(self, limit: int = -1) -> None:
        self._limit = limit

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        result = 0
        for msg in messages:
            result += len(msg.content)
            if 0 < self._limit < result:
                raise LLMTokenLimitException(
                    token_count=result, limit=self._limit, model="mock", llm_name=f"{self.__class__.__name__}"
                )
        return result


class MockLLMConfiguration(LLMConfigurationBase):
    def model_name(self) -> str:
        return self._model_name

    def __init__(self, model_name: str, timeout: int = 10, token_limit: int = -1) -> None:
        self._model_name = model_name
        self._timeout = timeout
        self._token_limit = token_limit


class MockLLM(LLMBase[MockLLMConfiguration]):
    def __init__(self, action: Optional[LLMMessagesToStr] = None, token_limit: int = -1) -> None:
        super().__init__(configuration=MockLLMConfiguration("mock"), token_counter=MockTokenCounter(token_limit))
        self._action = action

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        if self._action is not None:
            return LLMResult(choices=self._action(messages))
        return LLMResult(choices=[f"{self.__class__.__name__}"])

    @staticmethod
    def from_responses(responses: List[str]) -> MockLLM:
        return MockLLM(action=(lambda x: responses))

    @staticmethod
    def from_response(response: str) -> MockLLM:
        return MockLLM(action=(lambda x: [response]))

    @staticmethod
    def from_multi_line_response(responses: Iterable[str]) -> MockLLM:
        response = "\n".join(responses)
        return MockLLM(action=(lambda x: [response]))


class MockErrorLLM(LLMBase):
    def __init__(self, exception: LLMException = LLMException("From Mock", "mock")) -> None:
        super().__init__(configuration=MockLLMConfiguration("mock-error"))
        self.exception = exception

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        raise self.exception
