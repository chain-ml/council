import unittest
import council_ai


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("Hello World!", council_ai.hello_word())

    def test_hello_council(self):
        self.assertEqual("Hello Council!", council_ai.hello_word("Council"))
