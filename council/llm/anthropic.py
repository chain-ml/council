from __future__ import annotations

import abc
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence

from anthropic.types import Completion
from council.llm import LLMMessage


class AnthropicAPIClientResult:
    def __init__(self, choices: List[str], raw_response: Optional[Dict[str, Any]] = None) -> None:
        self._choices = choices
        self._raw_response = raw_response

    @property
    def choices(self) -> List[str]:
        return self._choices

    @property
    def raw_response(self) -> Optional[Dict[str, Any]]:
        return self._raw_response

    @staticmethod
    def from_completion(result: Completion) -> AnthropicAPIClientResult:
        """For legacy completion API"""
        return AnthropicAPIClientResult(choices=[result.completion])


class AnthropicAPIClientWrapper(ABC):

    @abc.abstractmethod
    def post_chat_request(self, messages: Sequence[LLMMessage]) -> AnthropicAPIClientResult:
        pass
