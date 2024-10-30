from __future__ import annotations

import abc
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence

from anthropic.types import Completion
from council.llm import LLMMessage


class Usage:
    """Represents token usage statistics for an Anthropic API request."""

    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_creation_prompt_tokens: int,
        cache_read_prompt_tokens: int,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cache_creation_prompt_tokens = cache_creation_prompt_tokens
        self.cache_read_prompt_tokens = cache_read_prompt_tokens
        self.total_tokens = cache_creation_prompt_tokens + cache_read_prompt_tokens + prompt_tokens + completion_tokens

    @staticmethod
    def from_dict(values: Dict[str, int]) -> Usage:
        prompt_tokens = values["input_tokens"]
        completion_tokens = values["output_tokens"]
        cache_creation_prompt_tokens = values.get("cache_creation_input_tokens", 0)
        cache_read_prompt_tokens = values.get("cache_read_input_tokens", 0)
        return Usage(prompt_tokens, completion_tokens, cache_creation_prompt_tokens, cache_read_prompt_tokens)

    @staticmethod
    def empty() -> Usage:
        return Usage(0, 0, 0, 0)


class AnthropicAPIClientResult:
    def __init__(self, choices: List[str], usage: Usage, raw_response: Optional[Dict[str, Any]] = None) -> None:
        self._choices = choices
        self._usage = usage
        self._raw_response = raw_response

    @property
    def choices(self) -> List[str]:
        return self._choices

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def raw_response(self) -> Optional[Dict[str, Any]]:
        return self._raw_response

    @staticmethod
    def from_completion(result: Completion) -> AnthropicAPIClientResult:
        """For legacy completion API"""
        return AnthropicAPIClientResult(choices=[result.completion], usage=Usage.empty())


class AnthropicAPIClientWrapper(ABC):

    @abc.abstractmethod
    def post_chat_request(self, messages: Sequence[LLMMessage]) -> AnthropicAPIClientResult:
        pass
