from __future__ import annotations

from typing import Any, Iterable, List, Optional, Protocol, Sequence

from council import LLMContext
from council.contexts import Consumption
from council.llm import (
    LLMBase,
    LLMConfigSpec,
    LLMConfigurationBase,
    LLMException,
    LLMMessage,
    LLMMessageTokenCounterBase,
    LLMResult,
    LLMTokenLimitException,
)


class LLMMessagesToStr(Protocol):
    def __call__(self, messages: Sequence[LLMMessage]) -> Sequence[str]: ...


class MockTokenCounter(LLMMessageTokenCounterBase):
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

    @classmethod
    def from_env(cls, *args: Any, **kwargs: Any) -> MockLLMConfiguration:
        raise NotImplementedError("MockLLMConfiguration doesn't support from_env() initialization.")

    @classmethod
    def from_spec(cls, spec: LLMConfigSpec) -> MockLLMConfiguration:
        raise NotImplementedError("MockLLMConfiguration doesn't support from_spec() initialization.")


class MockLLM(LLMBase[MockLLMConfiguration]):
    def __init__(self, action: Optional[LLMMessagesToStr] = None, token_limit: int = -1) -> None:
        super().__init__(configuration=MockLLMConfiguration("mock"), token_counter=MockTokenCounter(token_limit))
        self._action = action

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        choices = self._action(messages) if self._action is not None else [f"{self.__class__.__name__}"]
        return LLMResult(choices=choices, consumptions=[Consumption.call(1, "mock_llm")])

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
