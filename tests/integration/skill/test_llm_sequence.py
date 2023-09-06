import logging
import unittest
from typing import Any

import dotenv

from council.agents import Agent
from council.chains import Chain
from council.contexts import ChainContext, Budget, Consumption
from council.llm import AzureLLM
from council.runners import ParallelFor
from council.skills import LLMSkill


def book_title_generator(context: ChainContext) -> Any:
    result = context.try_last_message.map_or(lambda m: m.message, "")
    titles = result.split("\n")
    for t in titles:
        if len(t) > 5:
            yield t[2:]


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("council").setLevel(logging.DEBUG)

        dotenv.load_dotenv()
        self.llm = AzureLLM.from_env()

    def test_llm_sequence(self):
        system_prompt = "You are a Finance expert answering question about finance topic."
        defi_llm_skill = LLMSkill(llm=self.llm, system_prompt=system_prompt)
        system_prompt = "You are a Finance expert judging the answer of a candidate."
        rater_llm_skill = LLMSkill(llm=self.llm, system_prompt=system_prompt)

        chain = Chain("GPT-4", "Answer to an user prompt about Finance", [defi_llm_skill, rater_llm_skill])

        agent = Agent.from_chain(chain)
        result = agent.execute_from_user_message(message="What is inflation?")
        self.assertTrue(result.try_best_message.is_some())
        print(result.best_message)

    def test_llm_sequence_with_parallel_for(self):
        system_prompt = "Give the title of two must-read non-fiction books about a given topic"
        book_finder_llm_skill = LLMSkill(llm=self.llm, system_prompt=system_prompt)

        system_prompt = "Give five takeaways from a book"
        book_takeaways_llm_skill = LLMSkill(llm=self.llm, name="Take", system_prompt=system_prompt)

        chain = Chain(
            "GPT-Book",
            "...",
            [
                book_finder_llm_skill,
                ParallelFor(generator=book_title_generator, skill=book_takeaways_llm_skill, parallelism=4),
            ],
        )

        agent = Agent.from_chain(chain)
        limit_tokens = Consumption(1500, "token", "gpt-35-turbo")
        limit_calls = Consumption(3, "call", "LLMSkill")
        budget = Budget(6000, limits=[limit_tokens, limit_calls])
        result = agent.execute_from_user_message(message="corporate finance", budget=budget)

        self.assertTrue(result.try_best_message.is_some())
        self.assertLess(limit_tokens.value, 1500.0)
        self.assertTrue(budget.is_expired())
        print(result.best_message)
