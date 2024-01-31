import time
import unittest

from council.contexts import Budget, Consumption
from council.utils import OsEnviron


class TestBudget(unittest.TestCase):
    def test_default(self):
        with OsEnviron("COUNCIL_DEFAULT_BUDGET", None):
            b = Budget.default()
            self.assertEqual(30, b.duration)

    def test_remaining(self):
        with OsEnviron("COUNCIL_DEFAULT_BUDGET", None):
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
        with OsEnviron("COUNCIL_DEFAULT_BUDGET", 60):
            b = Budget.default()
            self.assertEqual(60, b.duration)

    def test_add_consumptions(self):
        first = Consumption(10, "first", "count")
        second = Consumption(20, "second", "count")
        budget = Budget(duration=10, limits=[first, second])
        self.assertFalse(budget.is_expired())

        consumptions = [Consumption(6, "first", "count"), Consumption(2, "second", "count")]
        budget.add_consumptions(consumptions)

        self.assertFalse(budget.is_expired())
        self.assertEqual(budget._remaining[0].value, 4)
        self.assertEqual(budget._remaining[1].value, 18)
