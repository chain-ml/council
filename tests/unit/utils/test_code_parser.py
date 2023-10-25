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

        self._message = "\n".join(
            ["Here is the code:"] + self._python1 + ["", "text", ""] + self._yaml + self._undefined + self._python2
        )

    def test_parse_all_python(self):
        code_blocs = CodeParser.extract_code_blocs(language="python", text=self._message)
        self.assertEqual(len(code_blocs), 2)
        self.assertTrue(all(code_bloc.is_language("python") for code_bloc in code_blocs))

    def test_parse_all(self):
        code_blocs = CodeParser.extract_code_blocs(text=self._message)
        self.assertEqual(4, len(code_blocs))

    def test_find_first(self):
        code_bloc = CodeParser.find_first(text=self._message)
        self.assertIsNotNone(code_bloc)
        self.assertEqual("\n".join(self._python1[1:-1]), code_bloc.code)

    def test_find_first_yaml(self):
        code_bloc = CodeParser.find_first(language="yaml", text=self._message)
        self.assertIsNotNone(code_bloc)
        self.assertEqual("\n".join(self._yaml[1:-1]), code_bloc.code)

    def test_find_first_not_exist(self):
        code_bloc = CodeParser.find_first(language="sh", text=self._message)
        self.assertIsNone(code_bloc)

    def test_find_last(self):
        code_bloc = CodeParser.find_last(text=self._message)
        self.assertIsNotNone(code_bloc)
        self.assertEqual("\n".join(self._python2[1:-1]), code_bloc.code)
