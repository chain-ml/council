from datetime import datetime, timezone

from unittest import TestCase

from council.contexts import ExecutionLogEntry


class TestExecutionLogEntry(TestCase):
    def test_logger_format(self):
        instance = ExecutionLogEntry("me")
        instance.log_info("a %s message", "test")

        self.assertEqual(instance._logs[0][2], "a test message")
        self.assertAlmostEquals(instance._logs[0][0].second, datetime.now(timezone.utc).second, delta=2)
