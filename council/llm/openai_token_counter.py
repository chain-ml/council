from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence

import tiktoken
from tiktoken import Encoding

from . import LLMessageTokenCounterBase, LLMMessage, LLMTokenLimitException

logger = logging.getLogger(__name__)


class TokenInfo:
    def __init__(self, *, tokens_limit: int, tokens_per_message: int, tokens_per_name: int) -> None:
        self.tokens_limit = tokens_limit
        self.tokens_per_message = tokens_per_message
        self.tokens_per_name = tokens_per_name

    @classmethod
    def for_model(cls, model: str) -> Optional[TokenInfo]:
        if model.startswith("gpt-3.5-turbo"):
            return cls._for_gpt_35_family(model)
        elif model.startswith("gpt-4"):
            return cls._for_gpt_4_family(model)
        elif model.startswith("gpt-4o"):
            return cls._for_gpt_4o_family(model)
        elif model.startswith("o1"):
            return cls._for_o1_family(model)

        return None

    @staticmethod
    def _for_gpt_35_family(model: str) -> TokenInfo:
        tokens_limit = 4_096 if model == "gpt-3.5-turbo-instruct" else 16_385
        return TokenInfo(tokens_limit=tokens_limit, tokens_per_message=3, tokens_per_name=1)

    @staticmethod
    def _for_gpt_4_family(model: str) -> TokenInfo:
        tokens_limit = 8_192 if model in ["gpt-4-0613", "gpt-4-0314"] else 128_000
        return TokenInfo(tokens_limit=tokens_limit, tokens_per_message=3, tokens_per_name=1)

    @staticmethod
    def _for_gpt_4o_family(model: str) -> TokenInfo:
        return TokenInfo(tokens_limit=128_000, tokens_per_message=3, tokens_per_name=1)

    @staticmethod
    def _for_o1_family(model: str) -> TokenInfo:
        return TokenInfo(tokens_limit=128_000, tokens_per_message=3, tokens_per_name=1)


class OpenAITokenCounter(LLMessageTokenCounterBase):
    """
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on
        how messages are converted to tokens.
        https://platform.openai.com/docs/models/overview for tokens
    """

    LATEST_ALIASES: Mapping[str, str] = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview": "gpt-4-0125-preview",
        "gpt-4": "gpt-4-0613",
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "o1-preview": "o1-preview-2024-09-12",
        "o1-mini": "o1-mini-2024-09-12",
    }

    def __init__(
        self, encoding: Encoding, model: str, limit: int = -1, tokens_per_message: int = 0, tokens_per_name: int = 0
    ) -> None:
        self._encoding = encoding
        self._model = model
        self._limit = limit
        self._tokens_per_message = tokens_per_message
        self._tokens_per_name = tokens_per_name

    def count_message_token(self, message: LLMMessage) -> int:
        num_tokens = self._tokens_per_message
        num_tokens += len(self._encoding.encode(message.content))
        num_tokens += len(self._encoding.encode(message.role.name))
        if message.name is not None:
            num_tokens += len(self._encoding.encode(message.name))
            num_tokens += self._tokens_per_name
        return num_tokens

    def count_messages_token(self, messages: Sequence[LLMMessage]) -> int:
        result = 0
        for message in messages:
            result += self.count_message_token(message)
        result += 3  # every reply is primed with <|start|>assistant<|message|>

        if 0 < self._limit < result:
            raise LLMTokenLimitException(token_count=result, limit=self._limit, model=self._model, llm_name=None)

        return result

    def filter_first_messages(self, messages: Sequence[LLMMessage], margin: int) -> List[LLMMessage]:
        """
        Filters the first messages from a list of LLM messages based on a token limit margin.

        Args:
            messages (Sequence[LLMMessage]): A list of LLMMessage objects representing the messages.
            margin (int): The token limit margin. The method will keep messages until the token count, including
                          assistant tokens, exceeds (limit + 3 - margin).

        Returns:
            List[LLMMessage]: A filtered list of LLMMessage objects representing the first messages.

        """
        count = 0
        limit = self._limit + 3 - margin
        if limit <= 0:
            return []

        result: List[LLMMessage] = []
        for message in reversed(messages):
            token = self.count_message_token(message)
            if count + token < limit:
                count += token
                result.insert(0, message)
            else:
                break
        return result

    def filter_last_messages(self, messages: Sequence[LLMMessage], margin: int) -> List[LLMMessage]:
        """
        Filters the last messages from a list of LLM messages based on a token limit margin.

        Args:
            messages (Sequence[LLMMessage]): A list of LLMMessage objects representing the messages.
            margin (int): The token limit margin. The method will keep messages until the token count, including
                          assistant tokens, exceeds (limit + 3 - margin).

        Returns:
            List[LLMMessage]: A filtered list of LLMMessage objects representing the first messages.

        """
        count = 0
        limit = self._limit + 3 - margin
        if limit <= 0:
            return []

        result: List[LLMMessage] = []
        for message in messages:
            token = self.count_message_token(message)
            if count + token < limit:
                count += token
                result.append(message)
            else:
                break
        return result

    @property
    def token_limit(self) -> int:
        return self._limit

    @staticmethod
    def from_model(model: str) -> Optional[OpenAITokenCounter]:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in OpenAITokenCounter.LATEST_ALIASES:
            return OpenAITokenCounter._return_alias(model, OpenAITokenCounter.LATEST_ALIASES[model])

        info = TokenInfo.for_model(model)
        if info is None:
            return None

        return OpenAITokenCounter(
            encoding,
            model=model,
            limit=info.tokens_limit,
            tokens_per_message=info.tokens_per_message,
            tokens_per_name=info.tokens_per_name,
        )

    @staticmethod
    def _return_alias(alias: str, last: str) -> Optional[OpenAITokenCounter]:
        logger.warning(f"{alias} may change over time. Returning num tokens assuming {last}.")
        return OpenAITokenCounter.from_model(model=last)
