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
        result = self._llm.post_chat_request(messages)
        if len(result.choices) < 1:
            return self._parse_line("")
        response = result.first_choice.lower()
        parsed = [self._parse_line(line) for line in response.split("\n") if line.strip().startswith("score")]

        return parsed[0]

    def _build_messages(self, message: ChatMessage) -> List[LLMMessage]:
        system_prompt = [
            "# Role:",
            "You are an assistant specialized in evaluating how similar an expected message and an actual message are.",
            "# Instructions:",
            "Compare the {expected} message and the {actual} message",
            "Give a similarity score out of 100%",
            "Unrelated messages have a 0% similarity score",
            "Provide the result exactly in the format `score: {similarity score} - short justification`",
        ]
        user_prompt = [
            "Please give the similarity score of the actual message compared to the expected one.",
            "Actual message:",
            message.message,
            "Expected message:",
            self._expected,
        ]

        result = [LLMMessage.system_message("\n".join(system_prompt)), LLMMessage.user_message("\n".join(user_prompt))]
        return result

    @staticmethod
    def _parse_line(line: str) -> float:
        line = line.lower().removeprefix("score").strip().replace("-", ":")
        try:
            score = line.split(":", 3)
            return float(score[1].strip(":% ")) / 100.0
        except Exception:
            raise
