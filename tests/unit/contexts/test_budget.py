import os
import time
import unittest

from council.contexts import Budget, Consumption


class TestBudget(unittest.TestCase):
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
        b.add_consumption(6, "unit", "test")
        b.add_consumption(50, "unit", "test2")
        self.assertFalse(b.is_expired())
        self.assertEquals(4, consumption.value)
        b.add_consumption(5, "unit", "test")
        self.assertTrue(b.is_expired())

    def test_consumption_expired(self):
        consumption = Consumption(10, "unit", "test")
        b = Budget(60, limits=[consumption])
        b.add_consumption(10, "unit", "test")
        self.assertFalse(b.is_expired())
        b.add_consumption(1, "unit", "test")
        self.assertTrue(b.is_expired())

    def test_can_consume(self):
        consumption = Consumption(10, "unit", "test")
        b = Budget(60, limits=[consumption])
        b.add_consumption(4, "unit", "test")

        self.assertTrue(b.can_consume(100, "integration", "test"))
        self.assertTrue(b.can_consume(1, "unit", "test"))
        self.assertTrue(b.can_consume(6, "unit", "test"))
        self.assertFalse(b.can_consume(7, "unit", "test"))

    def test_duration_expired(self):
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
