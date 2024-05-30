import json

from council.contexts import ChatMessage, SkillContext

from .. import SkillBase
from .google_context import GoogleSearchEngine


class GoogleSearchSkill(SkillBase):
    """
    A skill that performs a Google search.

    Notes:
        * GOOGLE_API_KEY environment variable needs to be set
        * GOOGLE_SEARCH_ENGINE_ID environment variable needs to be set

    """

    def __init__(self, nb_results: int = 5) -> None:
        super().__init__("gsearch")
        self.gs = GoogleSearchEngine.from_env()
        self.nb_results = nb_results

    def execute(self, context: SkillContext) -> ChatMessage:
        prompt = context.chat_history.try_last_user_message.unwrap("no user message")
        resp = self.gs.execute(query=prompt.message, nb_results=self.nb_results)  # type: ignore
        response_count = len(resp)
        if response_count > 0:
            return self.build_success_message(
                f"{self._name} {response_count} responses for {prompt.message}", json.dumps([r.dict() for r in resp])
            )
        return self.build_error_message("no response")
