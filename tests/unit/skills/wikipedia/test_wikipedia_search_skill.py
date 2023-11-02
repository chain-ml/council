import unittest

import yaml

from council import ChainContext, SkillContext
from council.skills.wikipedia import WikipediaPageSection, WikipediaSearchSkill
from council.utils import Option


class TestWikipediaSearchSkill(unittest.TestCase):
    def test_search(self):
        instance = WikipediaSearchSkill()
        chain_context = ChainContext.from_user_message("python programming language")
        context = SkillContext.from_chain_context(chain_context, Option.none())

        result = instance.execute(context)

        assert result.is_ok
        assert result.message.startswith("```yaml")
        assert result.message.endswith("```")

        values = yaml.safe_load(result.message[7:-3])
        page = WikipediaPageSection(**values[0])
        assert page.page_id == 23862
        assert page.title == "Python (programming language)"
