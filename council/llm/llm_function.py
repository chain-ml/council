from __future__ import annotations

from typing import Any, Generic, Iterable, List, Optional, Sequence, Union

from council.contexts import Consumption, LLMContext

from .llm_answer import LLMParsingException
from .llm_base import LLMBase
from .llm_message import LLMMessage, LLMMessageRole
from .llm_middleware import LLMMiddleware, LLMMiddlewareChain, LLMRequest, LLMResponse
from .llm_response_parser import LLMResponseParser, T_Response


class LLMFunctionResponse(Generic[T_Response]):
    """
    A class representing the response from an LLM function.

    This class wraps the LLM response along with a parsed response, providing additional
    access to response metadata like duration and consumptions.
    """

    def __init__(
        self, llm_response: LLMResponse, response: T_Response, previous_responses: Sequence[LLMResponse]
    ) -> None:
        self.llm_response = llm_response
        self._response = response
        self._previous_responses = list(previous_responses)

    @property
    def response(self) -> T_Response:
        """
        Get the parsed response.

        Returns:
            T_Response: The parsed response.
        """
        return self._response

    @property
    def duration(self) -> float:
        """
        Get the duration of the LLM function response.

        Returns:
            float: The time taken by the LLM function to produce the response, in seconds.
            This includes total duration of LLM function execution, including self-corrections if any.
        """
        return self.llm_response.duration + sum(response.duration for response in self._previous_responses)

    @property
    def consumptions(self) -> Sequence[Consumption]:
        """
        Get the consumptions associated with the LLM function response.

        Returns:
            Sequence[Consumption]: A sequence of consumption objects if available; otherwise, an empty sequence.
        """
        consumptions: List[Consumption] = []

        # consumptions of previous self-corrected responses if any
        for response in self._previous_responses:
            if response.result is not None:
                consumptions.extend(response.result.consumptions)

        # consumptions of the final response
        if self.llm_response.result is not None:
            consumptions.extend(self.llm_response.result.consumptions)

        return consumptions

    @staticmethod
    def from_llm_response(
        llm_response: LLMResponse, llm_response_parser: LLMResponseParser, previous_responses: Sequence[LLMResponse]
    ) -> LLMFunctionResponse:
        """
        Create an instance of LLMFunctionResponse from a raw LLM response and a parser function.

        Args:
            llm_response (LLMResponse): The raw response from the LLM.
            llm_response_parser (LLMResponseParser): A function that parses the LLM response into the desired format.
            previous_responses (Sequence[LLMResponse]): Prior LLM responses that could not be parsed successfully.

        Returns:
            LLMFunctionResponse: A new instance of LLMFunctionResponse containing the parsed response.
        """
        return LLMFunctionResponse(llm_response, llm_response_parser(llm_response), previous_responses)


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
        system_message: Optional[Union[str, LLMMessage]] = None,
        messages: Optional[Iterable[LLMMessage]] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initializes the LLMFunction with a middleware chain, response parser,
        system_message / messages, and retry settings.
        """

        self._llm_middleware = LLMMiddlewareChain(llm) if not isinstance(llm, LLMMiddlewareChain) else llm
        self._llm_config = self._llm_middleware.llm.configuration
        self._response_parser = response_parser
        self._max_retries = max_retries
        self._context = LLMContext.empty()
        self._messages = self._validate_messages(system_message, messages, LLMMessageRole.System)

    def _validate_messages(
        self,
        str_message: Optional[Union[str, LLMMessage]],
        messages: Optional[Iterable[LLMMessage]],
        role: LLMMessageRole,
    ) -> List[LLMMessage]:
        """Convert `str_message` and `messages` into a proper List[LLMMessage]"""
        if str_message is None and messages is None:
            raise ValueError("At least one of str message, messages is required")

        llm_messages: List[LLMMessage] = []
        if str_message is not None:
            llm_messages.append(self._build_llm_message(str_message, role))

        if messages is not None:
            llm_messages.extend(messages)

        return llm_messages

    @staticmethod
    def _build_llm_message(message: Union[str, LLMMessage], role: LLMMessageRole) -> LLMMessage:
        return message if isinstance(message, LLMMessage) else LLMMessage(role=role, content=message)

    def add_middleware(self, middleware: LLMMiddleware) -> None:
        self._llm_middleware.add_middleware(middleware)

    def execute_with_llm_response(
        self,
        user_message: Optional[Union[str, LLMMessage]] = None,
        messages: Optional[Iterable[LLMMessage]] = None,
        **kwargs: Any,
    ) -> LLMFunctionResponse[T_Response]:
        """
        Executes the LLM request with the provided user message and additional messages,
        handling errors and retries as configured.

        Args:
            user_message (Union[str, LLMMessage], optional): The primary message from the user or an LLMMessage object.
            messages (Iterable[LLMMessage], optional): Additional messages to include in the request.
            **kwargs: Additional keyword arguments to be passed to the LLMRequest.

        Returns:
            LLMFunctionResponse[T_Response]: The response from the LLM and the one processed by the response parser.

        Raises:
            FunctionOutOfRetryError: If all retry attempts fail, this exception is raised with details.
        """

        llm_messages: List[LLMMessage] = self._messages + self._validate_messages(
            user_message, messages, LLMMessageRole.User
        )
        new_messages: List[LLMMessage] = []
        exceptions: List[Exception] = []
        previous_responses: List[LLMResponse] = []

        retry = 0
        while retry <= self._max_retries:
            llm_messages = llm_messages + new_messages
            request = LLMRequest(context=self._context, messages=llm_messages, **kwargs)
            try:
                llm_response = self._llm_middleware.execute(request)
                return LLMFunctionResponse.from_llm_response(llm_response, self._response_parser, previous_responses)
            except LLMParsingException as e:
                exceptions.append(e)
                previous_responses.append(llm_response)
                new_messages = self._handle_error(e, llm_response, e.message)
            except LLMFunctionError as e:
                if not e.retryable:
                    raise e
                exceptions.append(e)
                previous_responses.append(llm_response)
                new_messages = self._handle_error(e, llm_response, e.message)
            except Exception as e:
                exceptions.append(e)
                previous_responses.append(llm_response)
                new_messages = self._handle_error(e, llm_response, f"Fix the following exception: `{e}`")

            retry += 1

        raise FunctionOutOfRetryError(self._max_retries, exceptions)

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
        return self.execute_with_llm_response(user_message, messages, **kwargs).response

    def _handle_error(self, e: Exception, response: LLMResponse, user_message: str) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        if response.result is None:
            self._context.logger.warning(f"Exception occurred: {error} without response.")
            return [LLMMessage.assistant_message("No response"), LLMMessage.user_message("Please retry.")]

        first_choice = response.result.first_choice
        error += f"\nResponse: {first_choice}"
        self._context.logger.warning(f"Exception occurred: {error} for response {first_choice}")
        return [LLMMessage.assistant_message(first_choice), LLMMessage.user_message(f"{user_message} Fix\n{error}")]
