import unittest

import dotenv

from council.agents import Agent
from council.controller import BasicController
from council.core import Chain, ChatHistory, AgentContext, Budget
from council.evaluator import BasicEvaluator
from council.llm import AzureConfiguration, AzureLLM
from council.skill.llm_skill import LLMSkill


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
