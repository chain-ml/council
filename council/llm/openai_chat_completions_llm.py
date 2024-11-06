from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import httpx
from council.contexts import Consumption, LLMContext
from council.llm import (
    ChatGPTConfigurationBase,
    LLMBase,
    LLMCallException,
    LLMConsumptionCalculatorBase,
    LLMCostCard,
    LLMCostManagerObject,
    LLMMessage,
    LLMMessageTokenCounterBase,
    LLMResult,
    TokenKind,
)
from council.utils.utils import DurationManager, truncate_dict_values_to_str


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


class Usage:
    """
    Represents token usage statistics for an OpenAI API request, with normalized token counting.

    This class extends the standard OpenAI usage stats by separately tracking reasoning and cached tokens
    while maintaining consistent total token counts. Unlike the OpenAI implementation, this class:
    - Subtracts reasoning_tokens from completion_tokens to avoid double-counting
    - Subtracts cached_tokens from prompt_tokens to avoid double-counting
    """

    def __init__(
        self, completion_tokens: int, prompt_tokens: int, total_tokens: int, reasoning_tokens: int, cached_tokens: int
    ) -> None:
        self._completion = completion_tokens
        self._prompt = prompt_tokens
        self._total = total_tokens
        self._reasoning = reasoning_tokens
        self._cached = cached_tokens

    def __str__(self) -> str:
        return f'prompt_tokens="{self._prompt}" total_tokens="{self._total}" completion_tokens="{self._completion}"'

    @property
    def prompt_tokens(self) -> int:
        """Number of tokens in the prompt, excluding cached tokens."""
        return self._prompt

    @property
    def completion_tokens(self) -> int:
        """Number of tokens in the completion, excluding reasoning tokens."""
        return self._completion

    @property
    def total_tokens(self) -> int:
        """Total number of tokens used (cached + prompt + reasoning + completion)."""
        return self._total

    @property
    def reasoning_tokens(self) -> int:
        """Number of reasoning completion tokens."""
        return self._reasoning

    @property
    def cached_tokens(self) -> int:
        """Number of cached prompt tokens."""
        return self._cached

    @staticmethod
    def from_dict(obj: Any) -> Usage:
        _completion_tokens = int(obj.get("completion_tokens"))
        _prompt_tokens = int(obj.get("prompt_tokens"))
        _total_tokens = int(obj.get("total_tokens"))

        completion_tokens_details = obj.get("completion_tokens_details")
        _reasoning_tokens = completion_tokens_details["reasoning_tokens"] if completion_tokens_details else 0
        if _reasoning_tokens > 0:
            _completion_tokens -= _reasoning_tokens

        prompt_tokens_details = obj.get("prompt_tokens_details")
        _cached_tokens = prompt_tokens_details["cached_tokens"] if prompt_tokens_details else 0
        if _cached_tokens > 0:
            _prompt_tokens -= _cached_tokens

        return Usage(_completion_tokens, _prompt_tokens, _total_tokens, _reasoning_tokens, _cached_tokens)


class OpenAIConsumptionCalculator(LLMConsumptionCalculatorBase):
    _cost_manager = LLMCostManagerObject.openai()
    COSTS_gpt_35_turbo_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_35_turbo_family")
    COSTS_gpt_4_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_4_family")
    COSTS_gpt_4o_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("gpt_4o_family")
    COSTS_o1_FAMILY: Mapping[str, LLMCostCard] = _cost_manager.get_cost_map("o1_family")

    def find_model_costs(self) -> Optional[LLMCostCard]:
        if self.model.startswith("o1"):
            return self.COSTS_o1_FAMILY.get(self.model)
        elif self.model.startswith("gpt-4o"):
            return self.COSTS_gpt_4o_FAMILY.get(self.model)
        elif self.model.startswith("gpt-4"):
            return self.COSTS_gpt_4_FAMILY.get(self.model)
        elif self.model.startswith("gpt-3.5-turbo"):
            return self.COSTS_gpt_35_turbo_FAMILY.get(self.model)

        return None

    def get_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        """
        Get consumptions specific for OpenAI:
            - 1 call
            - specified duration
            - cache_read_prompt, prompt, reasoning, completion and total tokens
            - corresponding costs if LLMCostCard can be found
        """
        consumptions = self.get_base_consumptions(duration, usage) + self.get_cost_consumptions(usage)
        return self.filter_zeros(consumptions)  # could occur for cache/reasoning tokens

    def get_base_consumptions(self, duration: float, usage: Usage) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.duration(duration, self.model),
            Consumption.token(usage.cached_tokens, self.format_kind(TokenKind.cache_read_prompt)),
            Consumption.token(usage.prompt_tokens, self.format_kind(TokenKind.prompt)),
            Consumption.token(usage.reasoning_tokens, self.format_kind(TokenKind.reasoning)),
            Consumption.token(usage.completion_tokens, self.format_kind(TokenKind.completion)),
            Consumption.token(usage.total_tokens, self.format_kind(TokenKind.total)),
        ]

    def get_cost_consumptions(self, usage: Usage) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        cached_tokens_cost = cost_card.input_cost(usage.cached_tokens) / 2
        prompt_tokens_cost = cost_card.input_cost(usage.prompt_tokens)
        reasoning_tokens_cost = cost_card.output_cost(usage.reasoning_tokens)
        completion_tokens_cost = cost_card.output_cost(usage.completion_tokens)
        total_cost = sum([cached_tokens_cost, prompt_tokens_cost, reasoning_tokens_cost, completion_tokens_cost])

        return [
            Consumption.cost(cached_tokens_cost, self.format_kind(TokenKind.cache_read_prompt, cost=True)),
            Consumption.cost(prompt_tokens_cost, self.format_kind(TokenKind.prompt, cost=True)),
            Consumption.cost(reasoning_tokens_cost, self.format_kind(TokenKind.reasoning, cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind(TokenKind.completion, cost=True)),
            Consumption.cost(total_cost, self.format_kind(TokenKind.total, cost=True)),
        ]


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
