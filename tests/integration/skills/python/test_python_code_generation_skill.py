import unittest

import dotenv

from council import DoWhile
from council.contexts import ChainContext, ChatHistory, ChatMessage, InfiniteBudget, SkillContext
from council.llm import AzureLLM, AzureLLMConfiguration
from council.runners import If, RunnerBase, Sequential, new_runner_executor
from council.skills.python.llm_helper import extract_code_block
from council.utils import Option

from council.skills.python import PythonCodeGenerationSkill, PythonCodeVerificationSkill


class TestPythonCodeGenSkill(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        self.llm = AzureLLM(AzureLLMConfiguration.from_env())

    def test_code_gen_say_hi(self):
        chat_history = ChatHistory.from_user_message("write code that says `Hi`")
        context = SkillContext.from_chain_context(ChainContext.from_chat_history(chat_history), Option.none())
        instance = PythonCodeGenerationSkill(self.llm)

        self.run_while(instance, context)
        result = context.last_message
        self.assertPythonCode('print("Hi")', result)

    def test_code_gen_function_and_update(self):
        chat_history = ChatHistory.from_user_message("write code that says `Hi {someone}!` to someone")
        context = SkillContext.from_chain_context(
            ChainContext.from_chat_history(chat_history, budget=InfiniteBudget()), Option.none()
        )
        code_template_lines = [
            "def function_name(name: str) -> str:",
            "# COUNCIL NO EDIT BEFORE THIS LINE",
            "    {code}",
            "",
            "# COUNCIL NO EDIT AFTER THIS LINE",
            "print(function_name('You'))",
        ]
        code_template = "\n".join(code_template_lines)

        instance = PythonCodeGenerationSkill(self.llm, code_template.format(code="pass"))
        verify = PythonCodeVerificationSkill(code_template)
        self.run_while(Sequential(instance, If(self.is_ok, verify)), context)
        result = context.last_message

        self.assertPythonCode(code_template.format(code="return f'Hi {name}!'"), result)

        chat_history.add_agent_message(result.message)
        chat_history.add_user_message("update the code to say Bye")
        context = SkillContext.from_chain_context(
            ChainContext.from_chat_history(chat_history, budget=InfiniteBudget()), Option.none()
        )

        self.run_while(Sequential(instance, If(self.is_ok, verify)), context)
        result = context.last_message
        self.assertPythonCode(code_template.format(code="return f'Bye {name}!'"), result)

    def test_code_gen_say_hi_with_useless_imports(self):
        chat_history = ChatHistory.from_user_message("write code that says `HeLLo! :D`")
        context = SkillContext.from_chain_context(ChainContext.from_chat_history(chat_history), Option.none())
        code_template_lines = [
            "import os",
            "import datetime",
            "import pandas",
            "# COUNCIL NO EDIT BEFORE THIS LINE",
            "",
            "{code}",
            "# COUNCIL NO EDIT AFTER THIS LINE",
            "import numpy as np",
        ]

        code_template = "\n".join(code_template_lines)

        instance = PythonCodeGenerationSkill(self.llm, code_template.format(code="# add your code here"))
        verify = PythonCodeVerificationSkill(code_template)
        self.run_while(Sequential(instance, If(self.is_ok, verify)), context)
        result = context.last_message

        self.assertPythonCode(code_template.format(code='print("HeLLo! :D")'), result)

    def assertPythonCode(self, expected: str, result: ChatMessage):
        python_code = extract_code_block(result.message, "python")
        python_code = PythonCodeVerificationSkill.normalize_code(python_code)
        expected = PythonCodeVerificationSkill.normalize_code(expected)
        self.assertEqual(python_code, expected)

    @staticmethod
    def is_ok(context: ChainContext) -> bool:
        return context.try_last_message.unwrap("last message").is_ok

    @staticmethod
    def while_predicate(context: ChainContext) -> bool:
        return (
            context.try_last_message.unwrap("last message").is_error
            and len([m for m in context.current.messages if m.is_error]) < 5
        )

    def run_while(self, skill: RunnerBase, context: ChainContext):
        runner = DoWhile(self.while_predicate, skill)
        runner.run(context, new_runner_executor())
