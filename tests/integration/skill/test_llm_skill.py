import unittest
from typing import List

import dotenv

from council.agents import Agent
from council.chains import Chain
from council.controllers import BasicController
from council.core import ChatHistory, AgentContext
from council.evaluators import BasicEvaluator
from council.llm import AzureConfiguration, AzureLLM, LLMMessage
from council.mocks import MockLLM
from council.runners import Budget
from council.skills.llm_skill import LLMSkill


def first_llm_message_content_to_str(messages: List[LLMMessage]) -> str:
    return messages[0].content


class TestLlmSkill(unittest.TestCase):
    """Requires the `AZURE` environment variables to be set"""

    def setUp(self) -> None:
        dotenv.load_dotenv()
        config = AzureConfiguration.from_env()
        llm = AzureLLM(config)
        system_prompt = "You are an agent developed with Council framework and you are a Finance expert."
        llm_skill = LLMSkill(llm=llm, system_prompt=system_prompt)

        controller = BasicController()
        evaluator = BasicEvaluator()
        chain = Chain("GPT-4", "Answer to an user prompt using gpt4", [llm_skill])
        self.agent = Agent(controller, [chain], evaluator)

    def test_basic_prompt(self):
        chat_history = ChatHistory.from_user_message(message="Hello, who are you?")
        run_context = AgentContext(chat_history)
        result = self.agent.execute(run_context, Budget(10))
        print(result.best_message)

    def test_template_prompt(self):
        llm = MockLLM(action=first_llm_message_content_to_str)
        llm_skill = LLMSkill(llm=llm, system_prompt="The last user message is: '{{chat_history.last_message}}'")
        agent = Agent.from_skill(llm_skill)
        result = agent.execute_from_user_message("User Message")

        self.assertEquals(result.best_message.message, "The last user message is: 'User Message'")
