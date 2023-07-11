import unittest

import dotenv

from council.core import ChainContext, Budget
from council.skill.google import GoogleNewsSkill, GoogleSearchSkill
from council.skill.google.google_context import GoogleNewsSearchEngine, GoogleSearchEngine
from council.utils.option import OptionException


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()

    def test_gnews(self):
        expected = 8
        gn = GoogleNewsSearchEngine(period="90d", suffix="Finance")
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEquals(len(resp), expected)

    def test_gsearch(self):
        expected = 8
        gn = GoogleSearchEngine.from_env()
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEquals(len(resp), expected)

    def test_gnews_skill(self):
        context = ChainContext.from_user_message("USD")
        context.chatHistory.add_user_message("EUR")

        skill = GoogleNewsSkill(suffix="Finance")
        result = skill.execute(context=context, budget=Budget(duration=10))
        self.assertTrue(result.is_ok())
        self.assertIn("EUR", result.message)

    def test_gsearch_skill(self):
        context = ChainContext.from_user_message("USD")

        skill = GoogleSearchSkill()
        result = skill.execute(context=context, budget=Budget(duration=10))
        self.assertTrue(result.is_ok())
        self.assertIn("USD", result.message)

    def test_skill_no_message(self):
        context = ChainContext.empty()
        skill = GoogleNewsSkill()
        try:
            _ = skill.execute(context=context, budget=Budget(duration=10))
            self.assertTrue(False)
        except OptionException as oe:
            self.assertTrue(oe)
