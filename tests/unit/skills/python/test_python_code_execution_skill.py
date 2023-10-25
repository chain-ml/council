import io
import pickle
import unittest

from council.contexts import ChainContext, ChatHistory, SkillContext
from council.utils import Option
from council.skills.python import PythonCodeExecutionSkill


class TestPythonCodeExecutionSkill(unittest.TestCase):
    def test_print_hi(self):
        user_message = "\n".join(["```python", 'print("hi!")', "```"])

        context = self._context_from_user_message(user_message)

        skill = PythonCodeExecutionSkill()
        result = skill.execute(context)

        self.assertTrue(result.is_ok)
        self.assertEqual("hi!\n", result.data["stdout"])

    def test_print_hi_env_var(self):
        user_message = "\n".join(
            ["```python", "import os", """print(f'hi {os.environ["TEST_PRINT_ENV_VAR"]}!')""", "```"]
        )

        context = self._context_from_user_message(user_message)

        env_var = {"TEST_PRINT_ENV_VAR": "you"}
        skill = PythonCodeExecutionSkill(env_var=env_var)
        result = skill.execute(context)

        self.assertTrue(result.is_ok)
        self.assertEqual("hi you!\n", result.data["stdout"])

    def test_pickle(self):
        user_message = "\n".join(
            [
                "```python",
                "import pickle",
                "import sys",
                'data =  {"text":"hi pickle!", "table": [i for i in range(4)]}',
                "pickle.dump(data, sys.stdout.buffer)",
                "```",
            ]
        )

        context = self._context_from_user_message(user_message)
        skill = PythonCodeExecutionSkill(decode_stdout=False)
        result = skill.execute(context)

        with io.BytesIO(result.data["stdout_bytes"]) as f:
            actual = pickle.load(f)
        self.assertEqual("hi pickle!", actual["text"])
        self.assertEqual([0, 1, 2, 3], actual["table"])

    def test_raise(self):
        user_message = "\n".join(
            ["```python", "print('hi!')", "raise Exception('this is an error')", "print('bye!')", "```"]
        )

        expected_stderr = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "<string>", line 2, in <module>',
                "Exception: this is an error",
                "",
            ]
        )
        context = self._context_from_user_message(user_message)

        skill = PythonCodeExecutionSkill()
        result = skill.execute(context)

        self.assertTrue(result.is_error)
        self.assertEqual("hi!\n", result.data["stdout"])
        self.assertEqual(expected_stderr, result.data["stderr"])

    @staticmethod
    def _context_from_user_message(user_message: str) -> SkillContext:
        chat_history = ChatHistory.from_user_message(user_message)
        return SkillContext.from_chain_context(ChainContext.from_chat_history(chat_history), Option.none())
