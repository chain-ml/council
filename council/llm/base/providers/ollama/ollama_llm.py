from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Union

from council.contexts import Consumption, LLMContext
from council.utils.utils import DurationManager
from ollama import Client
from ollama._types import Message, Options

from ...llm_base import LLMBase, LLMResult
from ...llm_message import LLMMessage
from .ollama_llm_configuration import OllamaLLMConfiguration
from .ollama_llm_cost import OllamaConsumptionCalculator


class OllamaLLM(LLMBase[OllamaLLMConfiguration]):
    def __init__(self, config: OllamaLLMConfiguration) -> None:
        """
        Initialize a new instance.

        Args:
            config (OllamaLLMConfiguration): configuration for the instance
        """
        super().__init__(name=f"{self.__class__.__name__}", configuration=config)

        self._client = Client()

    @property
    def client(self) -> Client:
        """
        Ollama Client.

        While self._post_chat_request() focuses on chat-based LLM interactions, you can use the client for broader
        model management, such as listing, pulling, and deleting models, generating completions and embeddings, etc.
        See https://github.com/ollama/ollama/blob/main/docs/api.md
        """

        return self._client

    def pull(self) -> Mapping[str, Any]:
        """Download the model from the ollama library."""
        return self.client.pull(model=self.model_name)

    def load(self, keep_alive: Optional[Union[float, str]] = None) -> Mapping[str, Any]:
        """Load LLM in memory."""
        keep_alive_value = keep_alive if keep_alive is not None else self._configuration.keep_alive_value
        return self.client.chat(model=self.model_name, messages=[], keep_alive=keep_alive_value)

    def unload(self) -> Mapping[str, Any]:
        """Unload LLM from memory."""
        return self.client.chat(model=self.model_name, messages=[], keep_alive=0)

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        messages_payload = self._build_messages_payload(messages)
        with DurationManager() as timer:
            response = self.client.chat(
                model=self.model_name,
                messages=messages_payload,
                stream=False,
                keep_alive=self._configuration.keep_alive_value,
                format=self._configuration.format,
                options=Options(**self._configuration.params_to_options()),  # type: ignore
            )

        return LLMResult(
            choices=self._to_choices(response),
            consumptions=self._to_consumptions(timer.duration, response),
            raw_response=dict(response),
        )

    @staticmethod
    def _build_messages_payload(messages: Sequence[LLMMessage]) -> List[Message]:
        return [Message(role=message.role.value, content=message.content) for message in messages]

    @staticmethod
    def _to_choices(response: Mapping[str, Any]) -> List[str]:
        return [response["message"]["content"]]

    @staticmethod
    def _to_consumptions(duration: float, response: Mapping[str, Any]) -> Sequence[Consumption]:
        calculator = OllamaConsumptionCalculator(response["model"])
        return calculator.get_consumptions(duration, response)
