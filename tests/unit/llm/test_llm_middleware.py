import unittest

from council.llm import LLMMessage, LLMException, LLMFallback
from council.llm.base.llm_exception import LLMOutOfRetriesException, LLMCallTimeoutException
from council.llm.llm_function.llm_middleware import (
    LLMRequest,
    LLMMiddlewareChain,
    LLMLoggingMiddleware,
    LLMRetryMiddleware,
)
from council.mocks import MockLLM, MockErrorLLM


class TestLlmMiddleware(unittest.TestCase):

    def setUp(self) -> None:
        self._llm = MockLLM.from_response("USD")

    def test_with_log(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        with_logs = LLMMiddlewareChain(self._llm)
        with_logs.add_middleware(LLMLoggingMiddleware())
        llm_response = with_logs.execute(request)
        result = llm_response.result.first_choice
        print(result)

    def test_with_retry(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        with_retry = LLMMiddlewareChain(MockErrorLLM())
        with_retry.add_middleware(LLMLoggingMiddleware())
        with_retry.add_middleware(LLMRetryMiddleware(retries=3, delay=1, exception_to_check=LLMException))
        with self.assertRaises(LLMOutOfRetriesException):
            _ = with_retry.execute(request)

    def test_with_no_retry(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        with_retry = LLMMiddlewareChain(MockErrorLLM())
        with_retry.add_middleware(LLMLoggingMiddleware())
        with_retry.add_middleware(LLMRetryMiddleware(retries=3, delay=1, exception_to_check=LLMCallTimeoutException))
        with self.assertRaises(LLMException):
            _ = with_retry.execute(request)

    def test_with_retry_no_error(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        with_retry = LLMMiddlewareChain(LLMFallback(MockErrorLLM(), MockLLM.from_response("USD")))
        with_retry.add_middleware(LLMLoggingMiddleware())
        with_retry.add_middleware(LLMRetryMiddleware(retries=3, delay=1, exception_to_check=LLMCallTimeoutException))
        response = with_retry.execute(request)
        self.assertEqual("USD", response.result.first_choice)
