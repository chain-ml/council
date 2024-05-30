import abc
from abc import ABC
from typing import List, Sequence

from council.llm import LLMMessage


class AnthropicAPIClientWrapper(ABC):

    @abc.abstractmethod
    def post_chat_request(self, messages: Sequence[LLMMessage]) -> List[str]:
        pass
