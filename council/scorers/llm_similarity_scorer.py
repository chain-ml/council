from typing import List, Dict, Any

from .scorer_base import ScorerBase
from council.contexts import ChatMessage
from council.llm import LLMBase, LLMMessage


class LLMSimilarityScorer(ScorerBase):
    """
    Using an LLM to compute a similarity score between two messages.
    """

    def __init__(self, llm: LLMBase, expected: str):
        """
        Initialize a new instance

        Parameters:
            llm (LLMBase): the LLM to be used
            expected (str): the expected text message
        """
        self._llm = llm
        self._expected = expected

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["expected"] = self._expected
        return result

    def _score(self, message: ChatMessage) -> float:
        messages = self._build_messages(message)
        choices = self._llm.post_chat_request(messages)
        if len(choices) < 1:
            return self._parse_response("")
        return self._parse_response(choices[0])

    def _build_messages(self, message: ChatMessage) -> List[LLMMessage]:
        prompt = [
            "You are an assistant specialized in evaluating the similarity between two messages.",
            "# Instructions",
            "# compare the {expected} message and the {actual} message",
            "# given a similarity score out of 100%",
            "# provide the result exactly in the format `score: {similarity score}`",
            "Expected message:",
            self._expected,
            "Actual message:",
            message.message,
        ]
        result = [LLMMessage.system_message("\n".join(prompt))]
        return result

    @staticmethod
    def _parse_response(llm_response: str) -> float:
        try:
            score = llm_response.strip().removeprefix("score")
            score = score.strip(":% ")
            return float(score) / 100.0
        except Exception:
            raise
