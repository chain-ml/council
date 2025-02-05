import glob
import os
import shutil
import tempfile
import time
import unittest

from council.llm import (
    LLMMessage,
    LLMException,
    LLMFallback,
    LLMOutOfRetriesException,
    LLMCallTimeoutException,
    LLMRequest,
    LLMMiddlewareChain,
    LLMLoggingMiddleware,
    LLMRetryMiddleware,
    LLMTimestampFileLoggingMiddleware,
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


class TestLlmTimestampFileLoggingMiddleware(unittest.TestCase):
    def setUp(self) -> None:
        self._llm = MockLLM.from_response("USD")
        self._llm_with_delay = MockLLM.from_response("USD", delay=1.0)
        self._temp_dir = tempfile.mkdtemp()

        self._log_prefix = "test_llm"

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir)

    def test_creates_separate_log_files(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        chain = LLMMiddlewareChain(self._llm)
        chain.add_middleware(LLMTimestampFileLoggingMiddleware(path=self._temp_dir, filename_prefix=self._log_prefix))

        chain.execute(request)
        time.sleep(1)  # ensure different timestamps

        chain.execute(request)

        log_files = glob.glob(os.path.join(self._temp_dir, f"{self._log_prefix}_*.log"))
        self.assertEqual(2, len(log_files))

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("LLM input", content)
                self.assertIn("LLM output", content)
                self.assertIn("USD", content)

    def test_with_delay(self):
        messages = [LLMMessage.user_message("Give me an example of a currency")]
        request = LLMRequest.default(messages)

        chain = LLMMiddlewareChain(self._llm_with_delay)
        chain.add_middleware(LLMTimestampFileLoggingMiddleware(path=self._temp_dir, filename_prefix=self._log_prefix))

        chain.execute(request)

        log_files = glob.glob(os.path.join(self._temp_dir, f"{self._log_prefix}_*.log"))
        self.assertEqual(1, len(log_files))

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("LLM input", content)
                self.assertIn("LLM output", content)
                self.assertIn("USD", content)
