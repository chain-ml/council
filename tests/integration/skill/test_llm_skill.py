import os
import unittest
from typing import List

import dotenv

from council.agents import Agent
from council.contexts import AgentContext
from council.llm import AzureLLM, LLMMessage
from council.mocks import MockLLM
from council.runners import Budget
from council.skills.llm_skill import LLMSkill


def first_llm_message_content_to_str(messages: List[LLMMessage]) -> List[str]:
    return [messages[0].content]


class TestLlmSkill(unittest.TestCase):
    """Requires the `AZURE` environment variables to be set"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM.from_env()
        system_prompt = "You are an agent developed with Council framework and you are a Finance expert."
        llm_skill = LLMSkill(llm=llm, system_prompt=system_prompt)
        self.agent = Agent.from_skill(llm_skill, "Answer to an user prompt about finance using gpt4")

    def test_basic_prompt(self):
        run_context = AgentContext.from_user_message("Hello, who are you?")
        result = self.agent.execute(run_context, Budget(10))

        self.assertTrue(result.try_best_message.is_some())
        print(result.best_message)

    def test_choices(self):
        os.environ["AZURE_LLM_N"] = "3"
        os.environ["AZURE_LLM_TEMPERATURE"] = "1.0"

        try:
            llm = AzureLLM.from_env()
            system_prompt = "You are an agent developed with Council framework and you are a Finance expert."
            llm_skill = LLMSkill(llm=llm, system_prompt=system_prompt)
            agent = Agent.from_skill(llm_skill, "Answer to an user prompt using gpt4")
            result = agent.execute_from_user_message("Give me examples of a currency", budget=Budget(6000))
            self.assertTrue(result.try_best_message.is_some())
            self.assertEquals(3, len(result.best_message.data.choices))

        finally:
            del os.environ["AZURE_LLM_N"]
            del os.environ["AZURE_LLM_TEMPERATURE"]

        self.assertEquals(os.getenv("AZURE_LLM_N"), None)
        self.assertEquals(os.getenv("AZURE_LLM_TEMPERATURE"), None)

    def test_template_prompt(self):
        llm = MockLLM(action=first_llm_message_content_to_str)
        llm_skill = LLMSkill(llm=llm, system_prompt="The last user message is: '{{chat_history.last_message}}'")
        agent = Agent.from_skill(llm_skill)
        result = agent.execute_from_user_message("User Message")

        self.assertTrue(result.try_best_message.is_some())
        self.assertEquals(result.best_message.message, "The last user message is: 'User Message'")
