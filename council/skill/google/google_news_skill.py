import json

from council.core import SkillBase, Budget
from council.core import ChainContext
from council.core.execution_context import SkillMessage
from .google_context import GoogleNewsSearchEngine


class GoogleNewsSkill(SkillBase):
    """
    A skill that performs a Google news search.

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
