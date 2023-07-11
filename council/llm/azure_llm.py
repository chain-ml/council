import logging
import httpx

from typing import List, Any

from .azure_configuration import AzureConfiguration
from .llm_message import LLMMessage
from .llm_exception import LLMException
from .llm_base import LLMBase


class AzureLLM(LLMBase):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: AzureConfiguration

    def __init__(self, config: AzureConfiguration):
        self.config = config

    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> str:
        payload = {
            "messages": [message.dict() for message in messages],
            "temperature": self.config.temperature,
        }
        for key, value in kwargs.items():
            payload[key] = value

        return self.post_request(payload, **kwargs)

    def post_request(self, payload, **kwargs: Any):
        uri = f"{self.config.api_base}/openai/deployments/{self.config.deployment_name}/chat/completions"
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        params = {"api-version": self.config.api_version}
        logging.debug(f'message="Sending request" payload="{payload}"')
        with httpx.Client() as client:
            client.timeout.read = self.config.timeout
            response = client.post(url=uri, headers=headers, params=params, json=payload)
            return AzureLLM.validate_and_parse_response(response)

    @staticmethod
    def validate_and_parse_response(response: httpx.Response) -> str:
        if response.status_code != httpx.codes.OK:
            raise LLMException(f"Wrong status code: {response.status_code}. Reason: {response.text}")
        return response.json()["choices"][0]["message"]["content"]
