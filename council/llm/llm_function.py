from typing import Any, Callable, Generic, List, Optional, Sequence, TypeVar

from council import LLMContext
from council.llm import LLMBase, LLMMessage, LLMParsingException
from council.llm.llm_middleware import LLMMiddlewareChain, LLMRequest

T_Response = TypeVar("T_Response")

LLMResponseParser = Callable[[str], T_Response]


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
    def __init__(self, llm: LLMBase, response_parser: LLMResponseParser, system_message: str) -> None:
        self._llm_middleware = LLMMiddlewareChain(llm)
        self._system_message = LLMMessage.system_message(system_message)
        self._response_parser = response_parser
        self._max_retries = 3
        self._context = LLMContext.empty()

    def execute(self, user_message: str, **kwargs: Any) -> T_Response:
        messages = [self._system_message, LLMMessage.user_message(user_message)]
        new_messages: List[LLMMessage] = []
        exceptions: List[Exception] = []

        retry = 0
        while retry <= self._max_retries:
            messages = messages + new_messages
            request = LLMRequest(context=self._context, messages=messages, **kwargs)
            try:
                llm_response = self._llm_middleware.execute(request)
                if llm_response.result is not None:
                    response = llm_response.result.first_choice
                    return self._response_parser(response)
            except LLMParsingException as e:
                exceptions.append(e)
                new_messages = self._handle_error(e, response, e.message)
            except LLMFunctionError as e:
                exceptions.append(e)
                if not e.retryable:
                    raise e
                new_messages = self._handle_error(e, response, e.message)
            except Exception as e:
                exceptions.append(e)
                new_messages = self._handle_error(e, response, f"Fix the following exception: `{e}`")

            retry += 1

        raise FunctionOutOfRetryError(self._max_retries, exceptions)

    def _handle_error(self, e: Exception, response: str, user_message: str) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        self._context.logger.warning(f"Exception occurred: {error} for response {response}")
        return [LLMMessage.assistant_message(response), LLMMessage.user_message(f"{user_message} Fix\n{error}")]
