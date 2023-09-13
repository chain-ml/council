import unittest

from council.contexts import Budget, Consumption, ExecutionLogEntry, MonitoredBudget


class TestMonitoredBudget(unittest.TestCase):
    def test_add_consumption(self):
        log_entry = ExecutionLogEntry("test")
        limits = [Consumption(10, "count", "first"), Consumption(20, "count", "second")]

        budget = MonitoredBudget(log_entry, Budget(60, limits))
        budget._add_consumption(Consumption(1, "count", "first"))

        self.assertEqual(log_entry._consumptions[0].value, 1)
        self.assertEqual(budget._remaining[0].value, 9)
