import unittest
from council.core.execution_context import ChatMessageBase, ChatMessageKind


class MyChatMessage(ChatMessageBase):
    pass


class TestChatMessageBase(unittest.TestCase):
    def test_str_short_message(self):
        short = MyChatMessage("this is a short message", ChatMessageKind.Agent)
        self.assertEqual("Message of kind short: this is a short message", f"{short}")

    def test_str_long_message_is_truncated(self):
        long = MyChatMessage("this is an extremely long message that is going to be truncated", ChatMessageKind.Agent)
        self.assertEqual("Message of kind long: this is an extremely long message that is going to...", f"{long}")
