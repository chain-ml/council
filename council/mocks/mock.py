from __future__ import annotations

from typing import List, Sequence

from council import ChatMessage
from council.contexts import Monitorable, Monitored, ScorerContext
from council.llm import LLMMessage
from council.scorers import ScorerBase


class MockMultipleResponses:
    def __init__(self, responses: List[List[str]]) -> None:
        self._count = 0
        self._responses = ["\n".join(resp) for resp in responses]

    def __call__(self, messages: Sequence[LLMMessage]) -> Sequence[str]:
        return self.call(messages)

    def call(self, _messages: Sequence[LLMMessage]) -> Sequence[str]:
        if self._count < len(self._responses):
            self._count += 1
        return [self._responses[self._count - 1]]


def llm_message_content_to_str(messages: Sequence[LLMMessage]) -> Sequence[str]:
    return [msg.content for msg in messages]


class MockErrorSimilarityScorer(ScorerBase):
    def __init__(self, exception: Exception = Exception()) -> None:
        super().__init__()
        self.exception = exception

    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        raise self.exception


class MockMonitored(Monitored):
    def __init__(self, name: str = "mock") -> None:
        super().__init__(name, Monitorable("mock"))
