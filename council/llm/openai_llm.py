import logging
import httpx

from typing import List, Any

from .openai_configuration import OpenAIConfiguration
from .llm_message import LLMMessage
from .llm_exception import LLMException
from .llm_base import LLMBase


# chat completion API for OpenAI - might rename for clarity
# doesn't yet support functions
class OpenAILLM(LLMBase):
    """
    Represents an OpenAI language model.
    """

    config: OpenAIConfiguration

    def __init__(self, config: OpenAIConfiguration):
        self.config = config

    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> str:
        payload = {"messages": [message.dict() for message in messages]}

        return self.post_request(payload, **kwargs)

    def post_request(self, payload, **kwargs: Any):
        uri = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": self.config.authorization, "Content-Type": "application/json"}
        self.set_default_values(payload)
        for key, value in kwargs.items():
            payload[key] = value

        logging.debug(f'message="Sending request" payload="{payload}"')
        with httpx.Client() as client:
            client.timeout.read = self.config.timeout
            response = client.post(url=uri, headers=headers, json=payload)
            return OpenAILLM.validate_and_parse_response(response)

    def set_default_values(self, payload):
        if self.config.model:
            payload.setdefault("model", self.config.model)
        if self.config.temperature:
            payload.setdefault("temperature", float(self.config.temperature))
        if self.config.top_p:
            payload.setdefault("top_p", float(self.config.top_p))
        if self.config.n:
            payload.setdefault("n", int(self.config.n))
        if self.config.presence_penalty:
            payload.setdefault("presence_penalty", float(self.config.presence_penalty))
        if self.config.frequency_penalty:
            payload.setdefault("frequency_penalty", float(self.config.frequency_penalty))

    @staticmethod
    def validate_and_parse_response(response: httpx.Response) -> str:
        if response.status_code != httpx.codes.OK:
            raise LLMException(f"Wrong status code: {response.status_code}. Reason: {response.text}")
        return response.json()["choices"][0]["message"]["content"]
