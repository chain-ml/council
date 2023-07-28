import unittest

from council.llm import LLMMessage, OpenAITokenCounter, LLMTokenLimitException


class TestLlmOpenAI(unittest.TestCase):
    def test_token_counter(self):
        model = "gpt-3.5-turbo"
        counter = OpenAITokenCounter.from_model(model)
        messages = self._get_messages()

        self.assertEqual(129, counter.count_messages_token(messages))

    def test_token_counter_exception(self):
        model = "gpt-4"
        counter = OpenAITokenCounter.from_model(model)
        messages = self._get_messages(70)
        with self.assertRaises(LLMTokenLimitException) as cm:
            counter.count_messages_token(messages)
        self.assertEqual(
            "token_count=8823 is exceeding model gpt-4-0613 limit of 8192 tokens.",
            str(cm.exception),
        )

    def test_first_message_filter(self):
        model = "gpt-3.5-turbo"
        counter = OpenAITokenCounter.from_model(model)
        messages = self._get_messages()

        filtered = counter.filter_first_messages(messages, 4000)
        self.assertEqual(4, len(filtered))
        self.assertEqual(messages[-1], filtered[-1])

    def test_last_message_filter(self):
        model = "gpt-3.5-turbo"
        counter = OpenAITokenCounter.from_model(model)
        messages = self._get_messages()

        filtered = counter.filter_last_messages(messages, 4000)
        self.assertEqual(4, len(filtered))
        self.assertEqual(messages[0], filtered[0])

    @staticmethod
    def _get_messages(repeat: int = 1):
        messages = [
            LLMMessage.system_message(
                "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."
            ),
            LLMMessage.system_message("New synergies will help drive top-line growth.", name="example_user"),
            LLMMessage.system_message("Things working well together will increase revenue.", name="example_assistant"),
            LLMMessage.system_message(
                "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
                name="example_user",
            ),
            LLMMessage.system_message(
                "Let's talk later when we're less busy about how to do better.", name="example_assistant"
            ),
            LLMMessage.user_message(
                "This late pivot means we don't have time to boil the ocean for the client deliverable."
            ),
        ]

        return messages * repeat
