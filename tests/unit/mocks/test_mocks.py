import unittest

from council.llm import LLMMessage
from council.mocks import MockLLM, llm_message_content_to_str


class TestPrompt(unittest.TestCase):
    def test_empty(self):
        m = MockLLM()
        r = m.post_chat_request([])
        self.assertEqual(r, "MockLLM")

    def test_from_response(self):
        m = MockLLM.from_response("Test")
        r = m.post_chat_request([])
        self.assertEqual(r, "Test")

    def test_from_responses(self):
        m = MockLLM.from_responses(["Test1", "Test2"])
        r = m.post_chat_request([])
        self.assertEqual(r, "Test1\nTest2")

    def test_from_message(self):
        m = MockLLM(action=llm_message_content_to_str)
        r = m.post_chat_request([LLMMessage.user_message("Test")])
        self.assertEqual(r, "Test")
