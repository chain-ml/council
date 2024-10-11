from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable, List, Optional, Protocol, Sequence

from council.contexts import LLMContext

from .llm_base import LLMBase, LLMMessage, LLMResult
from .llm_exception import LLMOutOfRetriesException


class LLMRequest:
    def __init__(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> None:
        self._context = context
        self._messages = messages
        self._kwargs = kwargs

    @property
    def context(self) -> LLMContext:
        return self._context

    @property
    def messages(self) -> Sequence[LLMMessage]:
        return self._messages

    @property
    def kwargs(self) -> Any:
        return self._kwargs

    @staticmethod
    def default(messages: Sequence[LLMMessage], **kwargs: Any) -> LLMRequest:
        """Creates a default LLMRequest with an empty context."""
        return LLMRequest(LLMContext.empty(), messages, **kwargs)


class LLMResponse:
    def __init__(self, request: LLMRequest, result: Optional[LLMResult], duration: float) -> None:
        self._request = request
        self._result = result
        self._duration = duration

    @property
    def result(self) -> Optional[LLMResult]:
        return self._result

    @property
    def value(self, default: str = "") -> str:
        return self._result.first_choice if self._result is not None else default

    @property
    def duration(self) -> float:
        return self._duration

    @staticmethod
    def empty(request: LLMRequest) -> LLMResponse:
        """Creates an empty LLMResponse for a given request."""
        return LLMResponse(request, None, -1.0)


ExecuteLLMRequest = Callable[[LLMRequest], LLMResponse]


class LLMMiddleware(Protocol):
    """
    Protocol for defining LLM middleware.

    Middleware can intercept and modify requests and responses between the client and the LLM, introducing custom logic.
    """

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse: ...


class LLMMiddlewareChain:
    """Manages a chain of LLM middlewares and executes requests through them."""

    def __init__(self, llm: LLMBase, middlewares: Optional[Sequence[LLMMiddleware]] = None) -> None:
        self._llm = llm
        self._middlewares: list[LLMMiddleware] = list(middlewares) if middlewares else []

    def add_middleware(self, middleware: LLMMiddleware) -> None:
        """Add middleware to a chain."""
        self._middlewares.append(middleware)

    def execute(self, request: LLMRequest) -> LLMResponse:
        """Execute middleware chain."""

        def execute_request(r: LLMRequest) -> LLMResponse:
            start = time.time()
            result = self._llm.post_chat_request(r.context, request.messages, **r.kwargs)
            return LLMResponse(request, result, time.time() - start)

        handler: ExecuteLLMRequest = execute_request
        for middleware in reversed(self._middlewares):
            handler = self._wrap_middleware(middleware, handler)
        return handler(request)

    @property
    def llm(self) -> LLMBase:
        return self._llm

    def _wrap_middleware(self, middleware: LLMMiddleware, handler: ExecuteLLMRequest) -> ExecuteLLMRequest:
        def wrapped(request: LLMRequest) -> LLMResponse:
            return middleware(self._llm, handler, request)

        return wrapped


class LLMLoggingMiddleware:
    """Middleware for logging LLM requests and responses."""

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        request.context.logger.info(
            f"Sending request with {len(request.messages)} message(s) to {llm.configuration.model_name()}"
        )
        response = execute(request)
        if response.result is not None:
            request.context.logger.info(f"Response: `{response.result.first_choice}` in {response.duration} seconds")
        else:
            request.context.logger.warning("No response")
        return response


class LLMFileLoggingMiddleware:
    """Middleware for logging LLM requests and responses into a file."""

    def __init__(self, log_file: str, component_name: str) -> None:
        """Initialize the middleware with the path to the log_file."""

        self.log_file = log_file
        self.component_name = component_name
        self._lock = Lock()

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        self._log_llm_request(request)
        response = execute(request)
        self._log_llm_response(response)
        return response

    def _log_llm_request(self, request: LLMRequest) -> None:
        messages_str = "\n\n".join(message.format() for message in request.messages)
        self._log(f"LLM input for {self.component_name}:\n{messages_str}")

    def _log_llm_response(self, response: LLMResponse) -> None:
        if response.result is None:
            self._log(f"LLM output for {self.component_name} is not available")
            return
        self._log(
            f"LLM output for {self.component_name} Duration: {response.duration:.2f} Output:\n"
            f"{response.result.first_choice}"
        )

    def _log(self, content: str) -> None:
        """Append `content` to a current log file"""

        with self._lock:  # ensure each write is done atomically in case of multi-threading
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(f"\n{content}")


class LLMRetryMiddleware:
    """
    Middleware for implementing retry logic for LLM requests.

    Attempts to retry failed requests a specified number of times with a delay between attempts.
    """

    def __init__(self, retries: int, delay: float, exception_to_check: Optional[type[Exception]] = None) -> None:
        self._retries = retries
        self._delay = delay
        self._exception_to_check = exception_to_check if exception_to_check else Exception

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        attempt = 0
        exceptions: List[Exception] = []
        while attempt < self._retries:
            try:
                return execute(request)
            except Exception as e:
                if not isinstance(e, self._exception_to_check):
                    raise
                exceptions.append(e)
                attempt += 1
                if attempt >= self._retries:
                    break
                time.sleep(self._delay)

        raise LLMOutOfRetriesException(llm_name=llm.model_name, retry_count=attempt, exceptions=exceptions)
