import unittest

import dotenv

import json

from council.contexts import ChainContext, Budget, ChatHistory, SkillContext
from council.skills.google import GoogleNewsSkill, GoogleSearchSkill
from council.skills.google.google_context import GoogleNewsSearchEngine, GoogleSearchEngine
from council.utils.option import Option, OptionException


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()

    def test_gnews(self):
        expected = 8
        gn = GoogleNewsSearchEngine(period="90d", suffix="Finance")
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEqual(len(resp), expected)

    def test_gsearch(self):
        expected = 8
        gn = GoogleSearchEngine.from_env()
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEqual(len(resp), expected)

    def test_gnews_skill(self):
        context = ChainContext.from_user_message("USD", Budget(duration=10))
        context.chat_history.add_user_message("EUR")

        skill = GoogleNewsSkill(suffix="Finance")
        result = skill.execute(SkillContext.from_chain_context(context, Option.none()))
        self.assertTrue(result.is_ok)
        self.assertIn("EUR", result.message)
        for d in json.loads(result.data):
            self.assertGreater(len(d["title"]), 0)
            self.assertGreater(len(d["url"]), 0)
            self.assertEqual(len(d["snippet"]), 0)
            self.assertIsNotNone(d["date"])

    def test_gsearch_skill(self):
        context = ChainContext.from_user_message("USD", budget=Budget(duration=10))

        skill = GoogleSearchSkill()
        result = skill.execute(SkillContext.from_chain_context(context, Option.none()))
        self.assertTrue(result.is_ok)
        self.assertIn("USD", result.message)
        print(result.data)
        for d in json.loads(result.data):
            self.assertGreater(len(d["title"]), 0)
            self.assertGreater(len(d["url"]), 0)
            self.assertGreater(len(d["snippet"]), 0)
            self.assertIsNone(d["date"])

    def test_skill_no_message(self):
        context = ChainContext.from_chat_history(ChatHistory(), Budget(duration=10))
        skill = GoogleNewsSkill()
        try:
            _ = skill.execute(SkillContext.from_chain_context(context, Option.none()))
            self.assertTrue(False)
        except OptionException as oe:
            self.assertTrue(oe)
