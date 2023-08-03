import os
import time
import unittest

from council.runners import Budget, Consumption


class TestAgentResult(unittest.TestCase):
    def test_default(self):
        b = Budget.default()
        self.assertEqual(30, b.duration)

    def test_remaining(self):
        b = Budget.default()
        time.sleep(0.3)
        self.assertTrue(b.remaining_duration < 30)

    def test_remaining_consumption(self):
        consumption = Consumption(10, "unit", "test")
        b = Budget(60, limits=[consumption])
        b.add_consumption(Consumption(6, "unit", "test"), "unit-test")
        b.add_consumption(Consumption(50, "unit", "test2"), "unit-test")
        self.assertFalse(b.is_expired())
        self.assertEquals(4, consumption.value)
        b.add_consumption(Consumption(5, "unit", "test"), "unit-test")
        self.assertTrue(b.is_expired())

    def test_expired(self):
        b = Budget(duration=0.1)
        time.sleep(0.3)
        self.assertTrue(b.is_expired())

    def test_default_env_variable(self):
        os.environ["COUNCIL_DEFAULT_BUDGET"] = "60"

        try:
            b = Budget.default()
            self.assertEqual(60, b.duration)
        finally:
            del os.environ["COUNCIL_DEFAULT_BUDGET"]

        self.assertEquals(None, os.getenv("COUNCIL_DEFAULT_BUDGET", None))
