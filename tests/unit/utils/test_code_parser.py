import unittest

from council.utils import CodeParser


class TestCodeParser(unittest.TestCase):
    def setUp(self):
        self._python1 = [
            "```python",
            "import pickle",
            "import sys",
            'data =  {"text":"hi pickle!", "table": [i for i in range(4)]}',
            "pickle.dump(data, sys.stdout.buffer)",
            "```",
        ]

        self._yaml = ["```yaml", "config", "  url: http://localhost", "```"]

        self._python2 = [
            "```python",
            "print('hi!')",
            "raise Exception('this is an error')",
            "print('bye!')",
            "```",
        ]

        self._undefined = [
            "```",
            "print_message('hi!');",
            "```",
        ]

        self._empty_with_whitespace = [
            "```python",
            "",
            "```",
        ]

        self._empty_no_whitespace = [
            "```python",
            "```",
        ]

        self._message = "\n".join(
            ["Here is the code:"] + self._python1 + ["", "text", ""] + self._yaml + self._undefined + self._python2
        )

        self._message_empty_with_whitespace = "\n".join(
            ["Here is an empty code block with whitespace:"] + self._empty_with_whitespace
        )

        self._message_empty_no_whitespace = "\n".join(
            ["Here is an empty code block with no whitespace:"] + self._empty_no_whitespace
        )

    def test_parse_all_python(self):
        code_blocks = CodeParser.extract_code_blocs(language="python", text=self._message)
        self.assertEqual(len(code_blocks), 2)
        self.assertTrue(all(code_block.is_language("python") for code_block in code_blocks))

    def test_parse_all(self):
        code_blocks = CodeParser.extract_code_blocs(text=self._message)
        self.assertEqual(4, len(code_blocks))

    def test_find_first(self):
        code_block = CodeParser.find_first(text=self._message)
        self.assertIsNotNone(code_block)
        self.assertEqual("\n".join(self._python1[1:-1]), code_block.code)

    def test_find_first_yaml(self):
        code_block = CodeParser.find_first(language="yaml", text=self._message)
        self.assertIsNotNone(code_block)
        self.assertEqual("\n".join(self._yaml[1:-1]), code_block.code)

    def test_find_first_not_exist(self):
        code_block = CodeParser.find_first(language="sh", text=self._message)
        self.assertIsNone(code_block)

    def test_find_last(self):
        code_block = CodeParser.find_last(text=self._message)
        self.assertIsNotNone(code_block)
        self.assertEqual("\n".join(self._python2[1:-1]), code_block.code)

    def test_empty_code_block_with_whitespace(self):
        code_block = CodeParser.find_first(language="python", text=self._message_empty_with_whitespace)
        self.assertEqual(code_block.code, "")

    def test_empty_code_block_no_whitespace(self):
        code_block = CodeParser.find_first(language="python", text=self._message_empty_no_whitespace)
        self.assertEqual(code_block.code, "")
