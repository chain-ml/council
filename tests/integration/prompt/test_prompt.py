import unittest

import dotenv

from council.agents import Agent
from council.chains import Chain
from council.llm import AzureLLM
from council.mocks import MockLLM
from council.prompt import PromptBuilder
from council.skills import LLMSkill, PromptToMessages

template = """
Provided answers by candidate:
{% set answers = chain_history.last_message.split('\n') %}
{%- for answer in answers %}
{{loop.index}}: {{answer}}
{%- endfor %}
"""


class TestPrompt(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM.from_env()
        system_prompt = """You are scoring geographical answers based on fact.
        The question asked to the candidate was: {{chat_history.last_message}}"""
        p = PromptToMessages(PromptBuilder(template))
        self.llm_skill = LLMSkill(llm=llm, system_prompt=system_prompt, context_messages=p.to_system_message)

    def test_injection(self):
        llm = MockLLM.from_multi_line_response(["São Paulo", "Lima", "London"])
        mock_llm_skill = LLMSkill(llm=llm, system_prompt="")

        chain = Chain("GPT-4", "Answer to an user prompt about geography", [mock_llm_skill, self.llm_skill])
        agent = Agent.from_chain(chain)

        result = agent.execute_from_user_message("what are the three largest cities in South America?")
        self.assertTrue(result.try_best_message.is_some())
        print(result.best_message)
