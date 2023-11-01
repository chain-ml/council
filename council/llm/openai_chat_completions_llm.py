import httpx

from typing import List, Any, Protocol, Sequence, Optional

from . import LLMConfigurationBase
from .llm_message import LLMMessage, LLMessageTokenCounterBase
from .llm_exception import LLMCallException
from .llm_base import LLMBase, LLMResult
from council.contexts import LLMContext, Consumption


class Provider(Protocol):
    def __call__(self, payload: dict[str, Any]) -> httpx.Response:
        ...


class Message:
    _role: str
    _content: str

    def __init__(self, role: str, content: str):
        self._content = content
        self._role = role

    @property
    def content(self) -> str:
        return self._content

    @staticmethod
    def from_dict(obj: Any) -> "Message":
        _role = str(obj.get("role"))
        _content = str(obj.get("content"))
        return Message(_role, _content)


class Choice:
    _index: int
    _finish_reason: str
    _message: Message

    def __init__(self, index: int, finish_reason: str, message: Message):
        self._index = index
        self._finish_reason = finish_reason
        self._message = message

    @property
    def message(self) -> Message:
        return self._message

    @staticmethod
    def from_dict(obj: Any) -> "Choice":
        _index = int(obj.get("index"))
        _finish_reason = str(obj.get("finish_reason"))
        _message = Message.from_dict(obj.get("message"))
        return Choice(_index, _finish_reason, _message)


class Usage:
    _completion: int
    _prompt: int
    _total: int

    def __init__(self, completion_tokens: int, prompt_tokens: int, total_tokens: int):
        self._completion = completion_tokens
        self._prompt = prompt_tokens
        self._total = total_tokens

    def __str__(self) -> str:
        return f'prompt_tokens="{self._prompt}" total_tokens="{self._total}" completion_tokens="{self._completion}"'

    @property
    def prompt_tokens(self) -> int:
        return self._prompt

    @property
    def completion_tokens(self) -> int:
        return self._completion

    @property
    def total_tokens(self) -> int:
        return self._total

    @staticmethod
    def from_dict(obj: Any) -> "Usage":
        _completion_tokens = int(obj.get("completion_tokens"))
        _prompt_tokens = int(obj.get("prompt_tokens"))
        _total_tokens = int(obj.get("total_tokens"))
        return Usage(_completion_tokens, _prompt_tokens, _total_tokens)


class OpenAIChatCompletionsResult:
    _id: str
    _object: str
    _created: int
    _model: str
    _choices: List[Choice]
    _usage: Usage

    def __init__(self, id: str, object: str, created: int, model: str, choices: List[Choice], usage: Usage):
        self._id = id
        self._object = object
        self._usage = usage
        self._model = model
        self._choices = choices
        self._created = created

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

    def to_consumptions(self) -> Sequence[Consumption]:
        return [
            Consumption(1, "call", f"{self.model}"),
            Consumption(self.usage.prompt_tokens, "token", f"{self.model}:prompt_tokens"),
            Consumption(self.usage.completion_tokens, "token", f"{self.model}:completion_tokens"),
            Consumption(self.usage.total_tokens, "token", f"{self.model}:total_tokens"),
        ]

    @staticmethod
    def from_dict(obj: Any) -> "OpenAIChatCompletionsResult":
        _id = str(obj.get("id"))
        _object = str(obj.get("object"))
        _created = int(obj.get("created"))
        _model = str(obj.get("model"))
        _choices = [Choice.from_dict(y) for y in obj.get("choices")]
        _usage = Usage.from_dict(obj.get("usage"))
        return OpenAIChatCompletionsResult(_id, _object, _created, _model, _choices, _usage)


class OpenAIChatCompletionsModel(LLMBase):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    config: LLMConfigurationBase

    def __init__(
        self, config: LLMConfigurationBase, provider: Provider, token_counter: Optional[LLMessageTokenCounterBase]
    ):
        super().__init__(token_counter)
        self.config = config
        self._provider = provider

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        payload = self.config.build_default_payload()
        payload["messages"] = [message.dict() for message in messages]
        for key, value in kwargs.items():
            payload[key] = value

        context.logger.debug(f'message="Sending chat GPT completions request" payload="{payload}"')
        r = self._post_request(payload)
        context.logger.debug(f'message="got chat GPT completions result" id="{r.id}" model="{r.model}" {r.usage}')
        return LLMResult(choices=[c.message.content for c in r.choices], consumptions=r.to_consumptions())

    def _post_request(self, payload) -> OpenAIChatCompletionsResult:
        response = self._provider.__call__(payload)
        if response.status_code != httpx.codes.OK:
            raise LLMCallException(response.status_code, response.text)

        r = OpenAIChatCompletionsResult.from_dict(response.json())
        return r
