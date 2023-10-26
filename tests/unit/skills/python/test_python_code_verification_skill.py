import unittest

from council.contexts import ChainContext, ChatHistory, InfiniteBudget, SkillContext
from council.utils import Option

from council.skills.python import PythonCodeVerificationSkill


class TestPythonCodeVerificationSkill(unittest.TestCase):
    def test(self):
        code_template = """
import os
        
def say_hi() -> str:
# COUNCIL NO EDIT BEFORE THIS LINE

    pass

# COUNCIL NO EDIT AFTER THIS LINE

print(say_hi())

"""

        user_message = "\n".join(["```python", code_template, "```"])

        chat_history = ChatHistory.from_user_message(user_message)
        context = SkillContext.from_chain_context(
            ChainContext.from_chat_history(chat_history, budget=InfiniteBudget()), Option.none()
        )

        instance = PythonCodeVerificationSkill(code_template)
        result = instance.execute(context)

        print(result.message)

    def test_normalize(self):
        code_template = "\n".join(["import os  ", " ", "", "# this comment should be removed", "print('hi')   "])

        result = PythonCodeVerificationSkill.normalize_code(code_template)

        expected = "\n".join(["import os", "print('hi')", ""])

        self.assertEqual(expected, result)
