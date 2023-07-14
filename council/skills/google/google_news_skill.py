import json

from council.contexts import ChainContext, SkillMessage
from council.runners import Budget
from .google_context import GoogleNewsSearchEngine
from .. import SkillBase


class GoogleNewsSkill(SkillBase):
    """
    A skill that performs a Google News search.

    """

    def __init__(self, suffix: str = ""):
        super().__init__("gnews")
        self.gn = GoogleNewsSearchEngine(period="90d", suffix=suffix)

    def execute(self, context: ChainContext, budget: Budget) -> SkillMessage:
        prompt = context.chatHistory.last_user_message().unwrap("no user message")

        resp = self.gn.execute(query=prompt.message, nb_results=5)
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"gnews {response_count} responses for {prompt.message}", json.dumps([r.dict() for r in resp])
            )
        return self.build_error_message("no response")
