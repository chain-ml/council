import unittest

import dotenv

from council.agent import Agent
from council.controller import BasicController
from council.core import Chain, ChatHistory, AgentContext, Budget
from council.evaluator import BasicEvaluator
from council.llm import AzureConfiguration, AzureLLM
from council.mocks import MockLLM
from council.prompt import PromptBuilder
from council.prompt.prompt_builder import PromptToMessages
from council.skill import LLMSkill

template = """
The question was: {{chat_history.last_message}}
Provided answers by candidate:
{% set answers = chain_history.last_message.split('\n') %}
{%- for answer in answers %}
{{loop.index}}: {{answer}}
{%- endfor %}
"""


class TestPrompt(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        config = AzureConfiguration.from_env()
        llm = AzureLLM(config)
        system_prompt = "You are scoring geographical answers based on fact."
        p = PromptToMessages(PromptBuilder(template))
        self.llm_skill = LLMSkill(llm=llm, system_prompt=system_prompt, context_messages=p.get_messages_from_prompt)

    def test_injection(self):
        llm = MockLLM(["São Paulo", "Lima", "London"])
        mock_llm_skill = LLMSkill(llm=llm, system_prompt="")

        chain = Chain("GPT-4", "Answer to an user prompt about geography", [mock_llm_skill, self.llm_skill])
        agent = Agent(BasicController(), [chain], BasicEvaluator())

        chat_history = ChatHistory.from_user_message(message="what are the three largest cities in South America?")
        run_context = AgentContext(chat_history)
        result = agent.execute(run_context, Budget(60))
        print(result.best_message)
