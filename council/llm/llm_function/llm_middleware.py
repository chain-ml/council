from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from enum import Enum
from threading import Lock
from typing import Any, Callable, List, Optional, Protocol, Sequence

from council.contexts import Consumption, ContextLogger, LLMContext
from council.llm.base import LLMBase, LLMMessage, LLMOutOfRetriesException, LLMResult, T_Configuration


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
    def request(self) -> LLMRequest:
        return self._request

    @property
    def result(self) -> LLMResult:
        if self._result is None:
            raise RuntimeError("LLMResult is None")
        return self._result

    @property
    def maybe_result(self) -> Optional[LLMResult]:
        return self._result

    @property
    def has_result(self) -> bool:
        return self._result is not None

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

    def to_messages(self) -> List[LLMMessage]:
        messages = list(self.request.messages)

        if not self.has_result:
            return messages

        return messages + [LLMMessage.assistant_message(self.result.first_choice)]


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


class LLMLoggingStrategy(str, Enum):
    """Defines logging strategies for LLM middleware."""

    Minimal = "minimal"
    """Basic request/response info without details"""

    MinimalWithConsumptions = "minimal_consumptions"
    """Basic info with consumption details"""

    Verbose = "verbose"
    """Full request/response content"""

    VerboseWithConsumptions = "verbose_consumptions"
    """Full request/response content with consumption details"""

    @property
    def is_minimal(self) -> bool:
        """Whether this strategy uses minimal logging."""
        return self in (LLMLoggingStrategy.Minimal, LLMLoggingStrategy.MinimalWithConsumptions)

    @property
    def is_verbose(self) -> bool:
        """Whether this strategy uses verbose logging."""
        return self in (LLMLoggingStrategy.Verbose, LLMLoggingStrategy.VerboseWithConsumptions)

    @property
    def has_consumptions(self) -> bool:
        """Whether this strategy includes consumption details."""
        return self in (LLMLoggingStrategy.MinimalWithConsumptions, LLMLoggingStrategy.VerboseWithConsumptions)


class LLMLoggingMiddlewareBase:
    """Base middleware class for logging LLM requests, responses and consumptions."""

    def __init__(self, strategy: LLMLoggingStrategy, component_name: Optional[str]) -> None:
        self.strategy = strategy
        self.component_name = component_name

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        name = self.component_name if self.component_name is not None else llm.configuration.model_name()

        self._log_llm_request(request, name)
        response = execute(request)
        self._log_llm_response(response, name)
        self._log_consumptions(response, name)

        return response

    def _log_llm_request(self, request: LLMRequest, name: str) -> None:
        self._log(self._format_llm_request(request, name))

    def _format_llm_request(self, request: LLMRequest, name: str) -> str:
        log_message_start = f"LLM input for {name}:"
        if self.strategy.is_minimal:
            return f"{log_message_start} {len(request.messages)} message(s)"
        return f"{log_message_start}\n" + "\n\n".join(message.format() for message in request.messages)

    def _log_llm_response(self, response: LLMResponse, name: str) -> None:
        self._log(self._format_llm_response(response, name))

    def _format_llm_response(self, response: LLMResponse, name: str) -> str:
        log_message_start = f"LLM output for {name}"
        if not response.has_result:
            return f"{log_message_start} is not available"

        log_message = (
            f"{log_message_start} received in {response.duration:.4f} seconds, "
            f"{len(response.result.choices)} choice(s) returned"
        )
        if self.strategy.is_minimal:
            return log_message
        return f"{log_message}:\n{response.result.first_choice}"

    def _log_consumptions(self, response: LLMResponse, name: str) -> None:
        if not response.has_result:
            return

        if self.strategy.has_consumptions:
            for consumption in response.result.consumptions:
                self._log(f"Consumption for {name}: {consumption}")

    def _log(self, content: str) -> None:
        """Abstract method to be implemented by subclasses for actual logging."""
        raise NotImplementedError()

    @staticmethod
    def append_to_file(lock: Lock, *, file_path: str, content: str) -> None:
        """Append content to a file atomically"""
        with lock:  # ensure each write is done atomically in case of multi-threading
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(f"\n{content}")


class LLMLoggingMiddleware(LLMLoggingMiddlewareBase):
    """Middleware for logging LLM requests, responses and consumptions to the context logger."""

    def __init__(
        self, strategy: LLMLoggingStrategy = LLMLoggingStrategy.Verbose, component_name: Optional[str] = None
    ) -> None:
        super().__init__(strategy, component_name)
        self.context_logger: Optional[ContextLogger] = None

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        self.context_logger = request.context.logger
        response = super().__call__(llm, execute, request)
        self.context_logger = None
        return response

    def _log(self, content: str) -> None:
        if self.context_logger is None:
            raise RuntimeError("Context logger not set - calling LLMLoggingMiddleware._log() outside of __call__()")

        self.context_logger.info(content)


class LLMFileLoggingMiddleware(LLMLoggingMiddlewareBase):
    """Middleware for logging LLM requests, responses and consumptions into a `log_file` by appending to it."""

    def __init__(
        self,
        log_file: str,
        strategy: LLMLoggingStrategy = LLMLoggingStrategy.Verbose,
        component_name: Optional[str] = None,
    ) -> None:
        super().__init__(strategy, component_name)
        self.log_file = log_file
        self._lock = Lock()

    def _log(self, content: str) -> None:
        """Append `content` to a current log file"""

        self.append_to_file(self._lock, file_path=self.log_file, content=content)


class LLMTimestampFileLoggingMiddleware(LLMLoggingMiddlewareBase):
    """Middleware for logging LLM requests, responses and consumptions into separate log file for each request."""

    def __init__(
        self,
        strategy: LLMLoggingStrategy = LLMLoggingStrategy.Verbose,
        *,
        path: str = ".",
        filename_prefix: Optional[str] = None,
        component_name: Optional[str] = None,
    ) -> None:
        super().__init__(strategy, component_name)
        self.prefix = f"{filename_prefix}_" if filename_prefix is not None else ""
        self.path = path
        self._lock = Lock()
        self._current_filename: Optional[str] = None

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._current_filename = os.path.join(self.path, f"{self.prefix}{timestamp}.log")
        response = super().__call__(llm, execute, request)
        self._current_filename = None
        return response

    def _log(self, content: str) -> None:
        """Write content to the current log file"""
        if self._current_filename is None:
            raise RuntimeError(
                "Current log filename not set - calling LLMTimestampFileLoggingMiddleware._log() outside of __call__()"
            )

        self.append_to_file(self._lock, file_path=self._current_filename, content=content)


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


class CacheEntry:
    """Represents a cached LLM response."""

    def __init__(self, response: LLMResponse, ttl: float) -> None:
        self.response = self._rebuild_response(response)
        self.timestamp = time.time()
        self.ttl = ttl

    @staticmethod
    def _rebuild_response(response: LLMResponse) -> LLMResponse:
        """Modify the response so it has zero duration and consumptions in 'cached_' units"""
        if response.has_result:
            cached_consumptions: List[Consumption] = [
                Consumption(value=consumption.value, unit=f"cached_{consumption.unit}", kind=consumption.kind)
                for consumption in response.result.consumptions
            ]
            cached_result = LLMResult(response.result.choices, cached_consumptions, response.result.raw_response)
        else:
            cached_result = None
        cached_response = LLMResponse(response.request, cached_result, 0)

        return cached_response

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp >= self.ttl

    def renew_lifetime(self) -> None:
        """Update the timestamp to extend the lifetime."""
        self.timestamp = time.time()


class LLMCachingMiddleware:
    """Middleware that caches LLM responses to avoid duplicate calls."""

    def __init__(self, ttl: float = 300.0, cache_limit_size: int = 10) -> None:
        """
        Initialize the caching middleware.

        Args:
            ttl: Sliding window time-to-live in seconds for cache entries (default: 5 mins)
            cache_limit_size: Cache limit size in cached entries (default: 10)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._ttl = ttl
        self._cache_limit_size = cache_limit_size

    def __call__(self, llm: LLMBase, execute: ExecuteLLMRequest, request: LLMRequest) -> LLMResponse:
        self._remove_expired()
        key = self.get_hash(request, llm.configuration)

        if key in self._cache:
            entry = self._cache[key]
            entry.renew_lifetime()  # renew on hit
            self._cache.move_to_end(key)  # move to most recent
            return entry.response

        response = execute(request)

        # cache the new response
        self._cache[key] = CacheEntry(response=response, ttl=self._ttl)
        self._cache.move_to_end(key)
        self._enforce_cache_limit()

        return response

    @staticmethod
    def get_hash(request: LLMRequest, configuration: T_Configuration) -> str:
        """Convert the request and LLM configuration to a hash with hashlib.sha256."""
        serialized = json.dumps(
            {
                "configuration": {key: str(value) for key, value in configuration.__dict__.items()},
                # TODO: request.context is not serialized
                "messages": [m.normalize() for m in request.messages],
                "kwargs": request.kwargs,
            },
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _remove_expired(self) -> None:
        """Remove all expired cache entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
        for key in expired_keys:
            del self._cache[key]

    def _enforce_cache_limit(self) -> None:
        """Remove oldest entries if cache size exceeds limit."""
        while len(self._cache) > self._cache_limit_size:
            self._cache.popitem(last=False)  # remove the first (oldest) item

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
