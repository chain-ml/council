import json
import unittest

import dotenv

from council.agent import Agent
from council.agent_tests import AgentTestSuite, AgentTestCase
from council.controller import LLMController
from council.core import Chain
from council.evaluator import LLMEvaluator
from council.llm import AzureLLM, AzureConfiguration
from council.scorer import LLMSimilarityScorer
from council.skill import LLMSkill


class TestTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        llm = AzureLLM(config=AzureConfiguration.from_env())

        prompt = "you are an assistant expert in Finance. When asked about something else, say you don't know"
        finance_skill = LLMSkill(llm=llm, system_prompt=prompt)
        finance_chain = Chain(name="finance", description="answer questions about Finance", runners=[finance_skill])

        game_prompt = "you are an expert in video games. When asked about something else, say you don't know"
        game_skill = LLMSkill(llm=llm, system_prompt=game_prompt)
        game_chain = Chain(name="game", description="answer questions about Video games", runners=[game_skill])

        fake_prompt = "you will provide an answer not related to the question"
        fake_skill = LLMSkill(llm=llm, system_prompt=fake_prompt)
        fake_chain = Chain(name="fake", description="Can answer all questions", runners=[fake_skill])

        controller = LLMController(llm=llm, response_threshold=5)
        evaluator = LLMEvaluator(llm=llm)

        self.llm = llm
        self.agent = Agent(controller, [finance_chain, game_chain, fake_chain], evaluator)

    def test_run(self):
        tests = [
            AgentTestCase(
                prompt="What is inflation?",
                scorers=[
                    LLMSimilarityScorer(
                        self.llm,
                        expected="Inflation is the rate at which the general level of prices"
                        + " for goods and services is rising, and, subsequently, purchasing power is falling",
                    )
                ],
            ),
            AgentTestCase(
                prompt="What are the most popular video games",
                scorers=[LLMSimilarityScorer(self.llm, expected="The most popular video games are: ...")],
            ),
            AgentTestCase(
                prompt="What are the most popular movies",
                scorers=[LLMSimilarityScorer(self.llm, expected="The most popular movies are ...")],
            ),
        ]

        suite = AgentTestSuite(test_cases=tests)
        result = suite.run(self.agent, show_progressbar=False)
        print(json.dumps(result.to_dict(), indent=2))

        self.assertAlmostEqual(0.75, result.results[0].scorer_results[0].score, delta=0.1)
        self.assertAlmostEqual(0.75, result.results[1].scorer_results[0].score, delta=0.1)
        self.assertAlmostEqual(0.0, result.results[2].scorer_results[0].score, delta=0.1)
