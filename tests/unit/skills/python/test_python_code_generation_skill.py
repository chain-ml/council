import unittest

from council import ChainContext, ChatMessage, SkillContext
from council.llm import LLMMessageRole
from council.mocks import MockLLM
from council.skills.python import PythonCodeGenerationSkill
from council.utils import Option


class TestPythonCodeGeneration(unittest.TestCase):
    def test_correction(self):
        user_message = "generate code that says 'hi'"
        llm_answer = "\n".join(["Here is the code that says 'hi':", "```", "print('hi')", "```"])

        correction = "cannot find a code block of type `python`"

        llm = MockLLM()
        instance = PythonCodeGenerationSkill(llm)

        chain_context = ChainContext.from_user_message(user_message)
        context = SkillContext.from_chain_context(chain_context, Option.none())
        context.append(ChatMessage.skill(llm_answer, source=instance.SKILL_NAME))
        context.append(ChatMessage.skill(correction, source="correction skill", is_error=True))
        actual = instance.build_messages(context)

        self.assertEqual(len(actual), 3)
        self.assertEqual(actual[0].content, user_message)
        self.assertEqual(actual[0].role, LLMMessageRole.User)
        self.assertEqual(actual[1].content, llm_answer)
        self.assertEqual(actual[1].role, LLMMessageRole.Assistant)
        self.assertTrue(correction in actual[2].content)
        self.assertEqual(actual[2].role, LLMMessageRole.User)
