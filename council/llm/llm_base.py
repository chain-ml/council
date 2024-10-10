import abc
from typing import Any, Dict, Final, Generic, Optional, Sequence, TypeVar

from council.contexts import Consumption, LLMContext, Monitorable

from .llm_message import LLMessageTokenCounterBase, LLMMessage

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
        token_counter: Optional[LLMessageTokenCounterBase] = None,
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
