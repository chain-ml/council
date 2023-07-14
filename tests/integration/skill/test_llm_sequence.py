import logging
import unittest
from typing import Any

import dotenv

from council.agents import Agent
from council.chains import Chain
from council.controllers import BasicController
from council.contexts import ChatHistory, AgentContext, ChainContext
from council.evaluators import BasicEvaluator
from council.llm import AzureLLM
from council.runners import Budget, ParallelFor
from council.skills import LLMSkill


def book_title_generator(context: ChainContext, _b: Budget) -> Any:
    result = context.last_message.map_or(lambda m: m.message, "")
    titles = result.split("\n")
    for t in titles:
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

        agent = Agent(BasicController(), [chain], BasicEvaluator())
        chat_history = ChatHistory.from_user_message(message="What is inflation?")
        run_context = AgentContext(chat_history)
        result = agent.execute(run_context, Budget(180))
        print(result.best_message)

    def test_llm_sequence_with_parallel_for(self):
        system_prompt = "Give three non-fiction books about a given topic"
        book_finder_llm_skill = LLMSkill(llm=self.llm, system_prompt=system_prompt)

        system_prompt = "Give five takeaways from a book"
        book_takeaways_llm_skill = LLMSkill(llm=self.llm, name="Take", system_prompt=system_prompt)

        chain = Chain(
            "GPT-4",
            "...",
            [
                book_finder_llm_skill,
                ParallelFor(generator=book_title_generator, skill=book_takeaways_llm_skill, parallelism=4),
            ],
        )

        agent = Agent(BasicController(), [chain], BasicEvaluator())
        chat_history = ChatHistory.from_user_message(message="corporate finance")
        run_context = AgentContext(chat_history)
        result = agent.execute(run_context, Budget(6000))
        print(result.best_message)
