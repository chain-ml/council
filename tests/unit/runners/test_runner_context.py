import time
import unittest

from council.contexts import ChainContext, ChatHistory, ChatMessage
from council.runners import RunnerContext, Budget


class RunnerContextTests(unittest.TestCase):
    def setUp(self) -> None:
        chat_history = ChatHistory.from_user_message("Hello")
        self.chain_context = ChainContext(chat_history, [])
        self.messages = [
            ChatMessage.skill("first"),
            ChatMessage.skill("second")
        ]

    def test_should_stop_on_budget_expired(self):
        context = RunnerContext(self.chain_context, Budget(.1))
        self.assertFalse(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertFalse(context.should_stop())

        time.sleep(.2)
        self.assertTrue(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertTrue(context.should_stop())

    def test_should_stop_on_cancellation_token(self):
        context = RunnerContext(self.chain_context, Budget(10))
        self.assertFalse(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertFalse(context.should_stop())

        context.cancellation_token.cancel()
        self.assertFalse(context.budget.is_expired())
        self.assertTrue(context.cancellation_token.cancelled)
        self.assertTrue(context.should_stop())

    def test_current_messages(self):
        context = RunnerContext(self.chain_context, Budget(10), messages=self.messages)

        self.assertEqual([m.message for m in self.messages], [m.message for m in context.messages])
        self.assertEqual([m.message for m in self.messages], [m.message for m in context.previous_messages])
        self.assertEqual([], context.current_messages)

        context.append(ChatMessage.skill("new"))

        self.assertEqual(["first", "second", "new"], [m.message for m in context.messages])
        self.assertEqual(["first", "second"], [m.message for m in context.previous_messages])
        self.assertEqual(["new"], [m.message for m in context.current_messages])

    def test_fork(self):
        context = RunnerContext(self.chain_context, Budget(10), messages=self.messages)
        context.append(ChatMessage.skill("new"))
        self.assertEqual(["new"], [m.message for m in context.current_messages])

        new_context = context.fork()

        self.assertEqual(["first", "second", "new"], [m.message for m in new_context.previous_messages])
        self.assertEqual([], new_context.current_messages)

    def test_merge(self):
        context = RunnerContext(self.chain_context, Budget(10), messages=self.messages)
        new_context = context.fork()
        new_context.append(ChatMessage.skill("new"))
        self.assertEqual(["first", "second"], [m.message for m in context.previous_messages])
        self.assertEqual([], [m.message for m in context.current_messages])
        self.assertEqual(["new"], [m.message for m in new_context.current_messages])

        context.merge([new_context])

        self.assertEqual(["first", "second"], [m.message for m in context.previous_messages])
        self.assertEqual(["new"], [m.message for m in context.current_messages])
