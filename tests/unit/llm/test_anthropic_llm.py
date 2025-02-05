import unittest

from council.llm import LLMMessage
from council.llm.base.providers.anthropic.anthropic_completion_llm import AnthropicCompletionLLM


class TestAnthropicLLM(unittest.TestCase):
    def test_messages(self):
        messages = [LLMMessage.user_message("Hello")]
        expected = """

Human: Hello

Assistant:"""

        actual = AnthropicCompletionLLM._to_anthropic_messages(messages)
        assert actual == expected

    def test_messages_with_follow_up(self):
        messages = [
            LLMMessage.user_message("Hello"),
            LLMMessage.assistant_message("World!"),
            LLMMessage.user_message("Bye"),
        ]

        expected = """

Human: Hello

Assistant: World!

Human: Bye

Assistant:"""

        actual = AnthropicCompletionLLM._to_anthropic_messages(messages)
        assert actual == expected

    def test_messages_with_system_prompt(self):
        messages = [LLMMessage.system_message("You are an assistant"), LLMMessage.user_message("Hi")]

        expected = """

Human: You are an assistant
Hi

Assistant:"""

        actual = AnthropicCompletionLLM._to_anthropic_messages(messages)
        assert actual == expected
