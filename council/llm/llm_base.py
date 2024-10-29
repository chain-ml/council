import abc
from typing import Any, Dict, Final, Generic, List, Optional, Sequence, Tuple, TypeVar

from council.contexts import Consumption, LLMContext, Monitorable

from .llm_message import LLMMessage, LLMMessageTokenCounterBase

_DEFAULT_TIMEOUT: Final[int] = 30


class LLMConfigurationBase(abc.ABC):

    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @property
    def default_timeout(self) -> int:
        return _DEFAULT_TIMEOUT


T_Configuration = TypeVar("T_Configuration", bound=LLMConfigurationBase)


class LLMResult:
    """
    Represents a response from the LLM
    """

    def __init__(
        self,
        choices: Sequence[str],
        consumptions: Optional[Sequence[Consumption]] = None,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._choices = list(choices)
        self._consumptions = list(consumptions) if consumptions is not None else []
        self._raw_response = raw_response if raw_response is not None else {}

    @property
    def first_choice(self) -> str:
        return self._choices[0]

    @property
    def choices(self) -> Sequence[str]:
        return self._choices

    @property
    def consumptions(self) -> Sequence[Consumption]:
        return self._consumptions

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response


class LLMBase(Generic[T_Configuration], Monitorable, abc.ABC):
    """
    Abstract base class representing a language model.
    """

    def __init__(
        self,
        configuration: T_Configuration,
        token_counter: Optional[LLMMessageTokenCounterBase] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or "llm")
        self._token_counter = token_counter
        self._name = name or f"llm_{self.__class__.__name__}"
        self._configuration = configuration

    @property
    def configuration(self) -> T_Configuration:
        return self._configuration

    @property
    def model_name(self) -> str:
        return self.configuration.model_name()

    def post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        """
        Sends a chat request to the language model.

        Parameters:
            context (LLMContext): a context to track execution metrics
            messages (Sequence[LLMMessage]): A list of LLMMessage objects representing the chat messages.
            **kwargs: Additional keyword arguments for the chat request.

        Returns:
            LLMResult: The response from the language model.

        Raises:
            LLMTokenLimitException: If messages exceed the maximum number of tokens.
            Exception: If an error occurs during the execution of the chat request.
        """

        if self._token_counter is not None:
            _ = self._token_counter.count_messages_token(messages=messages)

        context.logger.debug(f'message="starting execution of llm {self._name} request"')
        try:
            with context:
                result = self._post_chat_request(context, messages, **kwargs)
                context.budget.add_consumptions(result.consumptions)
                return result
        except Exception as e:
            context.logger.exception(f'message="failed execution of llm {self._name} request" exception="{e}" ')
            raise e
        finally:
            context.logger.debug(f'message="done execution of llm {self._name} request"')

    @abc.abstractmethod
    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:
        pass


class LLMCostCard:
    """LLM cost per million token"""

    def __init__(self, input: float, output: float) -> None:
        self._input = input
        self._output = output

    @property
    def input(self) -> float:
        return self._input

    @property
    def output(self) -> float:
        return self._output

    def __str__(self) -> str:
        return f"${self.input}/${self.output} per 1m tokens"

    def input_cost(self, tokens: int) -> float:
        return tokens * self.input / 1e6

    def output_cost(self, tokens: int) -> float:
        return tokens * self.output / 1e6

    def get_costs(self, prompt_tokens: int, completion_tokens: int) -> Tuple[float, float]:
        """Return tuple of (prompt_tokens_cost, completion_token_cost)"""
        return self.input_cost(prompt_tokens), self.output_cost(completion_tokens)


class LLMConsumptionCalculator(abc.ABC):
    """Helper class to manage LLM consumptions."""

    def __init__(self, model: str):
        self.model = model

    def format_kind(self, token_kind: str, cost: bool = False) -> str:
        """Format Consumption.kind - from 'prompt' to '{self.model}:prompt_tokens'"""
        options = [
            "prompt",
            "completion",
            "total",
            "reasoning",  # OpenAI o1
            "cache_creation_prompt",  # Anthropic prompt caching
            "cache_read_prompt",  # Anthropic & OpenAI prompt caching
        ]
        result = f"{self.model}:_"
        if token_kind not in options:
            raise ValueError(f"Unknown kind for LLMConsumptionCalculator; expected one of `{','.join(options)}`")

        result += f"{token_kind}_tokens"

        if cost:
            result += "_cost"

        return result

    def get_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        """
        Get default consumptions:
            - 1 call
            - prompt, completion and total tokens
            - cost for prompt, completion and total tokens if LLMCostCard can be found
        """
        return self.get_token_consumptions(prompt_tokens, completion_tokens) + self.get_cost_consumptions(
            prompt_tokens, completion_tokens
        )

    def get_token_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        return [
            Consumption.call(1, self.model),
            Consumption.token(prompt_tokens, self.format_kind("prompt")),
            Consumption.token(completion_tokens, self.format_kind("completion")),
            Consumption.token(prompt_tokens + completion_tokens, self.format_kind("total")),
        ]

    def get_cost_consumptions(self, prompt_tokens: int, completion_tokens: int) -> List[Consumption]:
        cost_card = self.find_model_costs()
        if cost_card is None:
            return []

        prompt_tokens_cost, completion_tokens_cost = cost_card.get_costs(prompt_tokens, completion_tokens)
        return [
            Consumption.cost(prompt_tokens_cost, self.format_kind("prompt", cost=True)),
            Consumption.cost(completion_tokens_cost, self.format_kind("completion", cost=True)),
            Consumption.cost(prompt_tokens_cost + completion_tokens_cost, self.format_kind("total", cost=True)),
        ]

    @abc.abstractmethod
    def find_model_costs(self) -> Optional[LLMCostCard]:
        """Get LLMCostCard for self to calculate cost consumptions."""
        pass

    @staticmethod
    def filter_zeros(consumptions: List[Consumption]) -> List[Consumption]:
        return list(filter(lambda consumption: consumption.value > 0, consumptions))
