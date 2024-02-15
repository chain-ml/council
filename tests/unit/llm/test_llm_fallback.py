import unittest

from council.contexts import LLMContext
from council.llm import LLMFallback, LLMCallException
from council.mocks import MockLLM, MockErrorLLM


class TestLLMFallBack(unittest.TestCase):
    def test_no_fallback(self):
        m = MockLLM.from_response("Test")
        fb = MockLLM.from_response("FallBack")

        fb_llm = LLMFallback(m, fb)

        r = fb_llm.post_chat_request(LLMContext.empty(), [])
        self.assertEqual("Test", r.first_choice)

    def test_no_fallback_retry_negative(self):
        m = MockLLM.from_response("Test")
        fb = MockLLM.from_response("FallBack")

        fb_llm = LLMFallback(m, fb, retry_before_fallback=-1)

        r = fb_llm.post_chat_request(LLMContext.empty(), [])
        self.assertEqual("Test", r.first_choice)

    def test_fallback_with_retryable_error(self):
        m = MockErrorLLM(exception=LLMCallException(503, "Service unavailable", "mock-503"))
        fb = MockLLM.from_response("FallBack")

        fb_llm = LLMFallback(m, fb, retry_before_fallback=3)

        r = fb_llm.post_chat_request(LLMContext.empty(), [])
        self.assertEqual("FallBack", r.first_choice)

    def test_fallback_with_non_retryable_error(self):
        m = MockErrorLLM(exception=LLMCallException(401, "Unauthorized", "mock-401"))
        fb = MockLLM.from_response("FallBack")

        fb_llm = LLMFallback(m, fb, retry_before_fallback=1000)

        r = fb_llm.post_chat_request(LLMContext.empty(), [])
        self.assertEqual("FallBack", r.first_choice)

    def test_error(self):
        m = MockErrorLLM(exception=LLMCallException(401, "Unauthorized", "mock-401"))
        fb = MockErrorLLM(exception=LLMCallException(403, "Forbidden", "mock-403"))

        fb_llm = LLMFallback(m, fb, retry_before_fallback=1000)

        with self.assertRaises(LLMCallException) as e:
            _r = fb_llm.post_chat_request(LLMContext.empty(), [])

        self.assertEqual(e.exception.code, 403)
        self.assertEqual(e.exception.__cause__.code, 401)
        self.assertIn("Wrong status code: 403", str(e.exception))
