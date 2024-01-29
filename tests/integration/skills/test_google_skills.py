import unittest
from datetime import datetime, timedelta

import dotenv

import json

from council.contexts import ChainContext, Budget, ChatHistory, SkillContext
from council.skills.google import GoogleNewsSkill, GoogleSearchSkill
from council.skills.google.google_context import GoogleNewsSearchEngine, GoogleSearchEngine
from council.utils.option import Option, OptionException


class TestBase(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()

    def test_gnews_engine(self):
        expected = 8
        gn = GoogleNewsSearchEngine(period="90d", suffix="Finance")
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEqual(len(resp), expected)

    def test_gsearch_engine(self):
        expected = 8
        gn = GoogleSearchEngine.from_env()
        resp = gn.execute(query="USD", nb_results=expected)
        self.assertEqual(len(resp), expected)

    def test_gnews_skill(self):
        context = ChainContext.from_user_message("USD", Budget(duration=10))
        context.chat_history.add_user_message("EUR")

        expected_result_count = 4
        skill = GoogleNewsSkill(suffix="Finance", nb_results=expected_result_count, period="15d")
        result = skill.execute(SkillContext.from_chain_context(context, Option.none()))
        self.assertTrue(result.is_ok)

        json_loads = json.loads(result.data)
        self.assertLessEqual(expected_result_count, len(json_loads))
        for d in json_loads:
            self.assertGreater(len(d["title"]), 0)
            self.assertGreater(len(d["url"]), 0)
            self.assertEqual(len(d["snippet"]), 0)
            self.assertTrue(is_within_period(d["date"], 15))

    def test_gnews_skill_range(self):
        context = ChainContext.from_user_message("launch of Space Shuttle Endeavour", Budget(duration=10))
        context.chat_history.add_user_message("STS-134")

        expected_result_count = 4
        skill = GoogleNewsSkill(
            suffix="Space",
            nb_results=expected_result_count,
            period=None,
            start=datetime(2011, 5, 1),
            end=datetime(2011, 5, 30),
        )
        result = skill.execute(SkillContext.from_chain_context(context, Option.none()))
        self.assertTrue(result.is_ok)

        json_loads = json.loads(result.data)
        self.assertLessEqual(expected_result_count, len(json_loads))
        for d in json_loads:
            self.assertGreater(len(d["title"]), 0)
            self.assertGreater(len(d["url"]), 0)
            self.assertEqual(len(d["snippet"]), 0)
            # self.assertTrue(is_within_period(d["date"], 15))

    def test_gsearch_skill(self):
        context = ChainContext.from_user_message("USD", budget=Budget(duration=10))

        expected_result_count = 7
        skill = GoogleSearchSkill(nb_results=expected_result_count)
        result = skill.execute(SkillContext.from_chain_context(context, Option.none()))

        self.assertTrue(result.is_ok)
        self.assertIn("USD", result.message)

        json_loads = json.loads(result.data)
        self.assertEqual(expected_result_count, len(json_loads))
        for d in json_loads:
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


def parse_relative_timestamp(timestamp: str) -> datetime:
    if "minute" in timestamp:
        minutes_ago = int(timestamp.split()[0])
        return datetime.now() - timedelta(minutes=minutes_ago)
    if "hour" in timestamp:
        hours_ago = int(timestamp.split()[0])
        return datetime.now() - timedelta(hours=hours_ago)
    if "day" in timestamp:
        days_ago = int(timestamp.split()[0])
        return datetime.now() - timedelta(days=days_ago)
    if "week" in timestamp:
        weeks_ago = int(timestamp.split()[0])
        return datetime.now() - timedelta(weeks=weeks_ago)
    else:
        raise ValueError("Unsupported relative timestamp format")


def is_within_period(result: str, period: int) -> bool:
    if "ago" in result:
        publication_date = parse_relative_timestamp(result)
    else:
        publication_date = datetime.strptime(result, "%Y-%m-%d")

    delta = datetime.now() - publication_date
    return delta <= timedelta(days=period)
