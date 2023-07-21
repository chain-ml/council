import os
import unittest

from council.runners import Budget


class TestAgentResult(unittest.TestCase):

    def test_default(self):
        b = Budget.default()
        self.assertEqual(30, b.duration)

    def test_default_env_variable(self):
        os.environ["COUNCIL_DEFAULT_BUDGET"] = "60"
        b = Budget.default()
        self.assertEqual(60, b.duration)

        del os.environ["COUNCIL_DEFAULT_BUDGET"]
        self.assertEquals(None, os.getenv("COUNCIL_DEFAULT_BUDGET", None))
