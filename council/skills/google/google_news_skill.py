from typing import Optional
from datetime import datetime

import json

from council.contexts import ChatMessage, SkillContext
from .google_context import GoogleNewsSearchEngine
from .. import SkillBase


class GoogleNewsSkill(SkillBase):
    """
    A skill that performs a Google News search.

    """

    def __init__(
        self,
        suffix: str = "",
        nb_results=5,
        period: Optional[str] = "90d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        super().__init__("gnews")
        self.gn = GoogleNewsSearchEngine(period=period, suffix=suffix, start=start, end=end)
        self.nb_results = nb_results

    def execute(self, context: SkillContext) -> ChatMessage:
        prompt = context.chat_history.try_last_user_message.unwrap("no user message")
        resp = self.gn.execute(query=prompt.message, nb_results=self.nb_results)
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"gnews {response_count} responses for {prompt.message}", json.dumps([r.dict() for r in resp])
            )
        return self.build_error_message("no response")
