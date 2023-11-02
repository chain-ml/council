import yaml

from council import ChatMessage, SkillContext
from council.skills import SkillBase
from .wikipedia_client import WikipediaClient


class WikipediaSearchSkill(SkillBase):
    """
    A Skill to search for pages on Wikipedia
    """

    def __init__(self, name: str = "WikipediaSearch"):
        """
        Initialize a new instance

        Args:
            name (str): name of the skill
        """

        super().__init__(name)
        self._client = WikipediaClient()

    def execute(self, context: SkillContext) -> ChatMessage:
        last_message = context.try_last_message.unwrap("last message")
        pages = self._client.search_pages_custom(last_message.message, 5)
        response = "\n".join(["```yaml", yaml.dump([p.to_dict() for p in pages]), "```"])
        return self.build_success_message(response)
