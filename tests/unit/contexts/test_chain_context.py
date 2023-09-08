import time
import unittest

from council.contexts import Budget, ChainContext, ChatMessage, AgentContext
from council.mocks import MockMonitored


class ChainContextTests(unittest.TestCase):
    def setUp(self) -> None:
        agent_context = AgentContext.empty()
        agent_context.new_iteration()
        self.chain_context = ChainContext.from_agent_context(agent_context, MockMonitored(), "a chain")
        self.messages = [ChatMessage.skill("first"), ChatMessage.skill("second")]

    def test_should_stop_on_budget_expired(self):
        context = self.chain_context.fork_for(MockMonitored(), Budget(0.1))
        self.assertFalse(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertFalse(context.should_stop())

        time.sleep(0.2)
        self.assertTrue(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertTrue(context.should_stop())

    def test_should_stop_on_cancellation_token(self):
        context = self.chain_context.fork_for(MockMonitored(), Budget(10))
        self.assertFalse(context.budget.is_expired())
        self.assertFalse(context.cancellation_token.cancelled)
        self.assertFalse(context.should_stop())

        context.cancellation_token.cancel()
        self.assertFalse(context.budget.is_expired())
        self.assertTrue(context.cancellation_token.cancelled)
        self.assertTrue(context.should_stop())

    def test_current_messages(self):
        self.chain_context.extend(self.messages)
        context = self.chain_context.fork_for(MockMonitored(), Budget(10))

        self.assertEqual([m.message for m in self.messages], [m.message for m in context.messages])
        self.assertEqual([m.message for m in self.messages], [m.message for m in context._previous_messages.messages])
        self.assertEqual([], context._current_messages.messages)

        context.append(ChatMessage.skill("new"))

        self.assertEqual(["first", "second", "new"], [m.message for m in context.messages])
        self.assertEqual(["first", "second"], [m.message for m in context._previous_messages.messages])
        self.assertEqual(["new"], [m.message for m in context._current_messages.messages])

    def test_fork(self):
        self.chain_context.extend(self.messages)
        context = self.chain_context.fork_for(MockMonitored())
        context.append(ChatMessage.skill("new"))
        self.assertEqual(["new"], [m.message for m in context._current_messages.messages])

        new_context = context.fork_for(MockMonitored())

        self.assertEqual(["first", "second", "new"], [m.message for m in new_context._previous_messages.messages])
        self.assertEqual([], new_context._current_messages.messages)

    def test_merge(self):
        self.chain_context.extend(self.messages)
        context = self.chain_context.fork_for(MockMonitored())
        new_context = context.fork_for(MockMonitored())
        new_context.append(ChatMessage.skill("new"))
        self.assertEqual(["first", "second"], [m.message for m in context._previous_messages.messages])
        self.assertEqual([], [m.message for m in context._current_messages.messages])
        self.assertEqual(["new"], [m.message for m in new_context._current_messages.messages])

        context.merge([new_context])

        self.assertEqual(["first", "second"], [m.message for m in context._previous_messages.messages])
        self.assertEqual(["new"], [m.message for m in context._current_messages.messages])
