from datetime import datetime, timezone

from unittest import TestCase

from council.contexts import ExecutionLogEntry


class TestExecutionLogEntry(TestCase):
    def test_logger_format(self):
        instance = ExecutionLogEntry("me", node=None)
        instance.log_info("a %s message", "test")

        self.assertEqual(instance._logs[0][2], "a test message")
        self.assertAlmostEquals(instance._logs[0][0].second, datetime.now(timezone.utc).second, delta=2)

    def test_logger_no_params(self):
        instance = ExecutionLogEntry("me", node=None)
        instance.log_info("a % message")

        self.assertEqual(instance._logs[0][2], "a % message")
        self.assertAlmostEquals(instance._logs[0][0].second, datetime.now(timezone.utc).second, delta=2)
