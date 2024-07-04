from typing import Any, Callable, Generic, Iterable, List, Optional, Sequence, TypeVar, Union

from council.contexts import LLMContext

from .llm_answer import LLMParsingException
from .llm_base import LLMBase, LLMMessage
from .llm_middleware import LLMMiddlewareChain, LLMRequest, LLMResponse

T_Response = TypeVar("T_Response")

LLMResponseParser = Callable[[LLMResponse], T_Response]


class LLMFunctionError(Exception):
    """
    Exception raised when an error occurs during the execution of a function.
    """

    def __init__(self, message: str, retryable: bool = False):
        """
        Initialize the FunctionError instance.
        """
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class FunctionOutOfRetryError(LLMFunctionError):
    """
    Exception raised when the maximum number of function execution retries is reached.
    Stores all previous exceptions raised during retry attempts.
    """

    def __init__(self, retry_count: int, exceptions: Optional[Sequence[Exception]] = None):
        """
        Initialize the FunctionOutOfRetryException instance.

        Args:
            retry_count (int): The number of retries attempted.
            exceptions (List[Exception]): List of exceptions raised during retry attempts.
        """
        super().__init__(f"Exceeded maximum retries after {retry_count} attempts")
        self.exceptions = exceptions if exceptions is not None else []

    def __str__(self) -> str:
        message = super().__str__()
        if self.exceptions:
            message += "\nPrevious exceptions:\n"
            for i, exception in enumerate(self.exceptions, start=1):
                message += f"{i}. {exception}\n"
        return message


class LLMFunction(Generic[T_Response]):
    def __init__(
        self, llm: LLMBase, response_parser: LLMResponseParser, system_message: str, max_retries: int = 3
    ) -> None:
        self._llm_middleware = LLMMiddlewareChain(llm)
        self._system_message = LLMMessage.system_message(system_message)
        self._response_parser = response_parser
        self._max_retries = max_retries
        self._context = LLMContext.empty()

    def execute(
        self, user_message: Union[str, LLMMessage], messages: Optional[Iterable[LLMMessage]] = None, **kwargs: Any
    ) -> T_Response:
        um = user_message if isinstance(user_message, LLMMessage) else LLMMessage.user_message(user_message)
        llm_messages = [self._system_message, um]
        if messages:
            llm_messages = llm_messages + list(messages)
        new_messages: List[LLMMessage] = []
        exceptions: List[Exception] = []

        retry = 0
        while retry <= self._max_retries:
            llm_messages = llm_messages + new_messages
            request = LLMRequest(context=self._context, messages=llm_messages, **kwargs)
            try:
                llm_response = self._llm_middleware.execute(request)
                return self._response_parser(llm_response)
            except LLMParsingException as e:
                exceptions.append(e)
                new_messages = self._handle_error(e, llm_response, e.message)
            except LLMFunctionError as e:
                exceptions.append(e)
                if not e.retryable:
                    raise e
                new_messages = self._handle_error(e, llm_response, e.message)
            except Exception as e:
                exceptions.append(e)
                new_messages = self._handle_error(e, llm_response, f"Fix the following exception: `{e}`")

            retry += 1

        raise FunctionOutOfRetryError(self._max_retries, exceptions)

    def _handle_error(self, e: Exception, response: LLMResponse, user_message: str) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        if response.result is None:
            self._context.logger.warning(f"Exception occurred: {error} without response.")
            return [LLMMessage.assistant_message("No response"), LLMMessage.user_message("Please retry.")]

        first_choice = response.result.first_choice
        error += f"\nResponse: {first_choice}"
        self._context.logger.warning(f"Exception occurred: {error} for response {first_choice}")
        return [LLMMessage.assistant_message(first_choice), LLMMessage.user_message(f"{user_message} Fix\n{error}")]
