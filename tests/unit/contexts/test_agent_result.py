import unittest

from council.agents import AgentResult
from council.contexts import ScoredAgentMessage, ChatMessageBase


class TestAgentResult(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = AgentResult(
            messages=[
                ScoredAgentMessage(ChatMessageBase.agent("second"), 0.8),
                ScoredAgentMessage(ChatMessageBase.agent("best"), 1.0),
            ]
        )

    def test_best(self):
        self.assertEqual("best", self.instance.best_message.message)

    def test_best_empty(self):
        with self.assertRaises(ValueError):
            _ = AgentResult().best_message

    def test_try_best(self):
        self.assertEqual("best", self.instance.try_best_message.unwrap("best").message)

    def test_try_empty(self):
        self.assertTrue(AgentResult().try_best_message.is_none(), "is none")
