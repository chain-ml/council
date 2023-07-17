import logging

import httpx

from typing import List, Any, Protocol

from . import LLMConfigurationBase
from .llm_message import LLMMessage
from .llm_exception import LLMException
from .llm_base import LLMBase

logger = logging.getLogger(__name__)


class Provider(Protocol):
    def __call__(self, payload: dict[str, Any]) -> httpx.Response:
        ...


class OpenAIChatCompletionsModel(LLMBase):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: LLMConfigurationBase

    def __init__(self, config: LLMConfigurationBase, provider: Provider):
        self.config = config
        self._provider = provider

    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> str:
        payload = self.config.build_default_payload()
        payload["messages"] = [message.dict() for message in messages]
        for key, value in kwargs.items():
            payload[key] = value

        logger.debug(f'message="Sending chat GPT completions request" payload="{payload}"')
        response = self._provider.__call__(payload)
        return self.validate_and_parse_response(response)

    def validate_and_parse_response(self, response: httpx.Response) -> str:
        if response.status_code != httpx.codes.OK:
            raise LLMException(f"Wrong status code: {response.status_code}. Reason: {response.text}")
        choices = response.json()["choices"]

        n = self.config.n.unwrap() if self.config.n.is_some() else 1
        if n <= 1:
            return choices[0]["message"]["content"]

        return "--- choices ---\n".join(choice["message"]["content"] for choice in choices)
