import unittest

from council.chains import Chain
from council.mocks import MockSkill


class TestChain(unittest.TestCase):
    def test_monitor_from_skill(self):
        chain = Chain("a chain", "a short description", [MockSkill("mock skill")])

        self.assertEqual(chain.monitor.children["runner"].name, "mock skill")
        self.assertEqual(chain.monitor.children["runner"].type, "MockSkill")

    def test_monitor_from_multiple_skills(self):
        chain = Chain("name", "description", [MockSkill("first"), MockSkill("second")])

        self.assertEqual(chain.monitor.children["runner"].type, "Sequential")
        self.assertEqual(chain.monitor.children["runner"].children["sequence[0]"].name, "first")
        self.assertEqual(chain.monitor.children["runner"].children["sequence[1]"].name, "second")
