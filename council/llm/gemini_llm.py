from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import google.generativeai as genai  # type: ignore
from council.contexts import Consumption, LLMContext
from council.llm import (
    GeminiLLMConfiguration,
    LLMBase,
    LLMConfigObject,
    LLMMessage,
    LLMMessageRole,
    LLMProviders,
    LLMResult,
)
from google.ai.generativelanguage import FileData
from google.ai.generativelanguage_v1 import HarmCategory  # type: ignore
from google.generativeai.types import HarmBlockThreshold  # type: ignore


class GeminiLLM(LLMBase[GeminiLLMConfiguration]):
    def __init__(self, config: GeminiLLMConfiguration) -> None:
        """
        Initialize a new instance.

        Args:
            config(GeminiLLMConfiguration): configuration for the instance
        """
        super().__init__(name=f"{self.__class__.__name__}", configuration=config)
        genai.configure(api_key=config.api_key.value)
        self._model = genai.GenerativeModel(
            config.model_name(),
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED,
            },
        )

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        history, last = self._to_chat_history(messages=messages)
        chat = self._model.start_chat(history=history)
        response = chat.send_message(last)
        return LLMResult(choices=[response.text], consumptions=self.to_consumptions())

    def to_consumptions(self) -> Sequence[Consumption]:
        model = self._configuration.model_name()
        return [
            Consumption(1, "call", f"{model}"),
        ]

    @staticmethod
    def from_env() -> GeminiLLM:
        """
        Helper function that create a new instance by getting the configuration from environment variables.

        Returns:
            GeminiLLM
        """

        return GeminiLLM(GeminiLLMConfiguration.from_env())

    @staticmethod
    def from_config(config_object: LLMConfigObject) -> GeminiLLM:
        provider = config_object.spec.provider
        if not provider.is_of_kind(LLMProviders.Gemini):
            raise ValueError(f"Invalid LLM provider, actual {provider}, expected {LLMProviders.Gemini}")

        config = GeminiLLMConfiguration.from_spec(config_object.spec)
        return GeminiLLM(config=config)

    @staticmethod
    def _to_chat_history(messages: Sequence[LLMMessage]) -> Tuple[List[Any], Any]:
        history = []
        for message in messages[:-1]:
            if message.is_of_role(LLMMessageRole.System):
                history.append({"role": "user", "parts": GeminiLLM._get_parts(message)})
                history.append({"role": "model", "parts": [{"text": "Understood"}]})
            elif message.is_of_role(LLMMessageRole.User):
                history.append({"role": "user", "parts": GeminiLLM._get_parts(message)})
            elif message.is_of_role(LLMMessageRole.Assistant):
                history.append({"role": "model", "parts": [{"text": message.content}]})

        last_msg = messages[-1]
        return history, {"role": "user", "parts": GeminiLLM._get_parts(last_msg)}

    @staticmethod
    def _get_parts(message: LLMMessage) -> List[Any]:
        parts: List[Any] = []
        if message.is_of_role(LLMMessageRole.System):
            parts.append({"text": f"System Prompt: {message.content}"})
        elif message.is_of_role(LLMMessageRole.User):
            parts.append({"text": message.content})
        elif message.is_of_role(LLMMessageRole.Assistant):
            parts.append({"text": message.content})

        for data in message.data:
            if data.is_url:
                fd = FileData({"mime_type": data.mime_type, "file_uri": data.content})
                parts.append({"file_data": fd})
            else:
                parts.append({"inline_data": {"mime_type": data.mime_type, "data": data.content}})
        return parts
