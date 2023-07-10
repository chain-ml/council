import unittest
import council_ai


class TestHello(unittest.TestCase):
    def test_hello(self):
        self.assertEqual("Hello World!", council_ai.hello_word())
