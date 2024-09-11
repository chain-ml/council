from typing import Any, Generic, Iterable, List, Optional, Sequence, Union

from council.contexts import LLMContext

from .llm_answer import LLMParsingException
from .llm_base import LLMBase, LLMMessage
from .llm_middleware import LLMMiddleware, LLMMiddlewareChain, LLMRequest, LLMResponse
from .llm_response_parser import LLMResponseParser, T_Response


class LLMFunctionError(Exception):
    """
    Exception raised when an error occurs during the execution of an LLMFunction.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
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

    def __init__(self, retry_count: int, exceptions: Optional[Sequence[Exception]] = None) -> None:
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
    """
    Represents a function that handles interactions with an LLM,
    including error handling and retries. It uses middleware to manage the requests and responses.
    """

    def __init__(
        self,
        llm: Union[LLMBase, LLMMiddlewareChain],
        response_parser: LLMResponseParser,
        system_message: str,
        max_retries: int = 3,
    ) -> None:
        """
        Initializes the LLMFunction with a middleware chain, response parser, system message, and retry settings.
        """

        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._llm_config = self._llm_middleware.llm.configuration
        self._system_message = LLMMessage.system_message(system_message)
        self._response_parser = response_parser
        self._max_retries = max_retries
        self._context = LLMContext.empty()

    def add_middleware(self, middleware: LLMMiddleware) -> None:
        self._llm_middleware.add_middleware(middleware)

    def execute(
        self,
        user_message: Optional[Union[str, LLMMessage]] = None,
        messages: Optional[Iterable[LLMMessage]] = None,
        **kwargs: Any,
    ) -> T_Response:
        """
        Executes the LLM request with the provided user message and additional messages,
        handling errors and retries as configured.

        Args:
            user_message (Union[str, LLMMessage], optional): The primary message from the user or an LLMMessage object.
            messages (Iterable[LLMMessage], optional): Additional messages to include in the request.
            **kwargs: Additional keyword arguments to be passed to the LLMRequest.

        Returns:
            T_Response: The response from the LLM after processing by the response parser.

        Raises:
            FunctionOutOfRetryError: If all retry attempts fail, this exception is raised with details.
        """
        if user_message is None and messages is None:
            raise ValueError("At least one of 'user_message', 'messages' is required for LLMFunction.execute")

        llm_messages: List[LLMMessage] = [self._system_message]
        if user_message:
            um = user_message if isinstance(user_message, LLMMessage) else LLMMessage.user_message(user_message)
            llm_messages.append(um)
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
                if not e.retryable:
                    raise e
                exceptions.append(e)
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
