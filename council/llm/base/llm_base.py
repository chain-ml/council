import abc
from typing import Any, Dict, Final, Generic, Optional, Sequence, Type, TypeVar, get_args, get_origin

from council.contexts import Consumption, LLMContext, Monitorable
from typing_extensions import Self

from .llm_config_object import LLMConfigObject, LLMConfigSpec
from .llm_message import LLMMessage, LLMMessageTokenCounterBase

_DEFAULT_TIMEOUT: Final[int] = 30


class LLMConfigurationBase(abc.ABC):

    @abc.abstractmethod
    def model_name(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_env(cls, *args: Any, **kwargs: Any) -> Self:
        pass

    @classmethod
    @abc.abstractmethod
    def from_spec(cls, spec: LLMConfigSpec) -> Self:
        pass

    @property
    def default_timeout(self) -> int:
        return _DEFAULT_TIMEOUT


T_Configuration = TypeVar("T_Configuration", bound=LLMConfigurationBase)


class LLMResult:
    """
    Represents a result of an LLM call.
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
        """First choice, e.g. LLM response."""
        return self._choices[0]

    @property
    def choices(self) -> Sequence[str]:
        """List of LLM responses."""
        return self._choices

    @property
    def consumptions(self) -> Sequence[Consumption]:
        """List of consumptions associated with LLM call."""
        return self._consumptions

    @property
    def raw_response(self) -> Dict[str, Any]:
        """Raw response from LLM provider API."""
        return self._raw_response


class LLMBase(Generic[T_Configuration], Monitorable, abc.ABC):
    """
    Abstract base class representing chat LLM.
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

    @classmethod
    def _get_configuration_class(cls) -> Type[T_Configuration]:
        """
        Infers and returns the configuration class type from the generic argument
        to enable from_env() and from_config().
        """
        for base in getattr(cls, "__orig_bases__", []):
            if get_origin(base) is LLMBase:
                args = get_args(base)
                if args and issubclass(args[0], LLMConfigurationBase):
                    return args[0]

        raise NotImplementedError(
            "Could not automatically determine the configuration class type. "
            "Ensure the subclass is properly annotated with a specific LLMConfiguration."
        )

    @classmethod
    def from_env(cls) -> Self:
        config_class = cls._get_configuration_class()
        config = config_class.from_env()
        return cls(config)

    @classmethod
    def from_config(cls, config_object: LLMConfigObject) -> Self:
        config_class = cls._get_configuration_class()
        config = config_class.from_spec(config_object.spec)
        return cls(config)
