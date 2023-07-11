import unittest
import council


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("Hello World!", council.hello_word())

    def test_hello_council(self):
        self.assertEqual("Hello Council!", council.hello_word("Council"))
