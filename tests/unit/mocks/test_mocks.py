import unittest

from council.contexts import LLMContext
from council.llm import LLMMessage, LLMTokenLimitException
from council.mocks import MockLLM, llm_message_content_to_str


class TestPrompt(unittest.TestCase):
    def test_empty(self):
        m = MockLLM()
        r = m.post_chat_request(LLMContext.new_fake(), [])
        self.assertEqual("MockLLM", r.first_choice)

    def test_from_response(self):
        m = MockLLM.from_response("Test")
        r = m.post_chat_request(LLMContext.new_fake(), [])
        self.assertEqual("Test", r.first_choice)

    def test_from_multi_line_responses(self):
        m = MockLLM.from_multi_line_response(["Test1", "Test2"])
        r = m.post_chat_request(LLMContext.new_fake(), [])
        self.assertEqual("Test1\nTest2", r.first_choice)

    def test_from_message(self):
        m = MockLLM(action=llm_message_content_to_str)
        r = m.post_chat_request(LLMContext.new_fake(), [LLMMessage.user_message("Test")])
        self.assertEqual("Test", r.first_choice)

    def test_token_limit(self):
        m = MockLLM(action=llm_message_content_to_str, token_limit=3)
        with self.assertRaises(LLMTokenLimitException):
            _ = m.post_chat_request(LLMContext.new_fake(), [LLMMessage.user_message("Test")])
