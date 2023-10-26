import unittest

from council import ChainContext, ChatMessage, SkillContext
from council.mocks import MockLLM, MockMultipleResponses
from council.runners import RunnerExecutor
from council.skills.python import (
    build_code_generation_loop,
    PythonCodeExecutionSkill,
    PythonCodeVerificationSkill,
    PythonCodeGenerationSkill,
)
from council.utils import Option


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.good_code = ["Here is the code to say hi", "```python", "print('hi')", "```"]
        self.no_code = ["There is no code in this response"]
        self.bad_code = ["Here is the code that does not parse", "```python", "print('hi'", "```"]
        self.runtime_error = [
            "Here is the code that raises an exception",
            "```python",
            "raise Exception('this is an error')",
            "```",
        ]

    def test_success(self):
        last_message = self._run_up_to_n_iterations(10)
        self.assertTrue(last_message.is_ok)
        self.assertEqual("hi\n", last_message.data["stdout"])

    def test_no_code(self):
        last_message = self._run_up_to_n_iterations(1)
        self.assertTrue(last_message.is_error)
        self.assertTrue("code block" in last_message.message)

    def test_bad_code(self):
        last_message = self._run_up_to_n_iterations(2)
        self.assertTrue(last_message.is_error)
        self.assertEqual("SyntaxError: unexpected EOF while parsing (<unknown>, line 1)", last_message.message)

    def test_runtime_error(self):
        last_message = self._run_up_to_n_iterations(3)
        self.assertTrue(last_message.is_error)
        self.assertTrue("this is an error" in last_message.data["stderr"])

    def _run_up_to_n_iterations(self, count: int) -> ChatMessage:
        llm_responses = [self.no_code, self.bad_code, self.runtime_error, self.good_code]
        llm = MockLLM(action=MockMultipleResponses(responses=llm_responses))

        code_generation = PythonCodeGenerationSkill(llm)
        code_verification = PythonCodeVerificationSkill()
        code_execution = PythonCodeExecutionSkill()

        chain_context = ChainContext.from_user_message("generate python code")
        context = SkillContext.from_chain_context(chain_context, Option.none())

        instance = build_code_generation_loop(code_generation, code_verification, code_execution, count)
        instance.run(context, RunnerExecutor())
        return context.try_last_message.unwrap("last message")
