from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

import httpx
from council.contexts import Consumption, LLMContext
from council.utils.utils import DurationManager, truncate_dict_values_to_str

from ...llm_base import LLMBase, LLMResult
from ...llm_exception import LLMCallException
from ...llm_message import LLMMessage, LLMMessageTokenCounterBase
from .chat_gpt_configuration import ChatGPTConfigurationBase
from .openai_llm_cost import OpenAIConsumptionCalculator, Usage


class Provider(Protocol):
    def __call__(self, payload: dict[str, Any]) -> httpx.Response: ...


class Message:
    def __init__(self, role: str, content: str) -> None:
        self._content = content
        self._role = role

    @property
    def content(self) -> str:
        return self._content

    @staticmethod
    def from_dict(obj: Any) -> Message:
        _role = str(obj.get("role"))
        _content = str(obj.get("content"))
        return Message(_role, _content)


class Choice:
    def __init__(self, index: int, finish_reason: str, message: Message) -> None:
        self._index = index
        self._finish_reason = finish_reason
        self._message = message

    @property
    def message(self) -> Message:
        return self._message

    @staticmethod
    def from_dict(obj: Any) -> Choice:
        _index = int(obj.get("index"))
        _finish_reason = str(obj.get("finish_reason"))
        _message = Message.from_dict(obj.get("message"))
        return Choice(_index, _finish_reason, _message)


class OpenAIChatCompletionsResult:
    def __init__(
        self,
        id: str,
        object: str,
        created: int,
        model: str,
        choices: List[Choice],
        usage: Usage,
        raw_response: Dict[str, Any],
    ) -> None:
        self._id = id
        self._object = object
        self._usage = usage
        self._model = model
        self._choices = choices
        self._created = created

        self._raw_response = raw_response

    @property
    def id(self) -> str:
        return self._id

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def choices(self) -> Sequence[Choice]:
        return self._choices

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response

    def to_consumptions(self, duration: float) -> Sequence[Consumption]:
        consumption_calculator = OpenAIConsumptionCalculator(self.model)
        return consumption_calculator.get_consumptions(duration, self.usage)

    @staticmethod
    def from_response(response: Dict[str, Any]) -> OpenAIChatCompletionsResult:
        _id = str(response.get("id"))
        _object = str(response.get("object"))
        _created = int(response.get("created", -1))
        _model = str(response.get("model"))
        _choices = [Choice.from_dict(y) for y in response.get("choices", [])]
        _usage = Usage.from_dict(response.get("usage"))
        return OpenAIChatCompletionsResult(_id, _object, _created, _model, _choices, _usage, response)


class OpenAIChatCompletionsModel(LLMBase[ChatGPTConfigurationBase]):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(
        self,
        config: ChatGPTConfigurationBase,
        provider: Provider,
        token_counter: Optional[LLMMessageTokenCounterBase],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(configuration=config, token_counter=token_counter, name=name)
        self._provider = provider

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:

        payload = self._build_payload(messages)
        for key, value in kwargs.items():
            payload[key] = value

        context.logger.debug(
            f'message="Sending chat GPT completions request to {self._name}" payload="{truncate_dict_values_to_str(payload, 100)}"'
        )
        with DurationManager() as timer:
            r = self._post_request(payload)
        context.logger.debug(
            f'message="Got chat GPT completions result from {self._name}" id="{r.id}" model="{r.model}" {r.usage}'
        )
        return LLMResult(
            choices=[c.message.content for c in r.choices],
            consumptions=r.to_consumptions(timer.duration),
            raw_response=r.raw_response,
        )

    def _post_request(self, payload) -> OpenAIChatCompletionsResult:
        response = self._provider.__call__(payload)
        if response.status_code != httpx.codes.OK:
            raise LLMCallException(response.status_code, response.text, self._name)

        return OpenAIChatCompletionsResult.from_response(response.json())

    def _build_payload(self, messages: Sequence[LLMMessage]) -> Dict[str, Any]:
        payload = self._configuration.build_default_payload()
        msgs = []
        for message in messages:
            content: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]
            result: Dict[str, Any] = {"role": message.role.value}
            if message.name is not None:
                result["name"] = message.name
            for data in message.data:
                if data.is_image:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{data.mime_type};base64,{data.content}"}}
                    )
                elif data.is_url:
                    content.append({"type": "image_url", "image_url": {"url": f"{data.content}"}})
            result["content"] = content
            msgs.append(result)
        payload["messages"] = msgs
        return payload
