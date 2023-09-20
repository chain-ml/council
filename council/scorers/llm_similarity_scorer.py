from typing import List, Dict, Any

from .scorer_base import ScorerBase
from council.contexts import ChatMessage, ScorerContext
from council.llm import LLMBase, LLMMessage, MonitoredLLM


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
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._expected = expected
        self._system_message = self._build_system_prompt()

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["expected"] = self._expected
        return result

    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        messages = self._build_messages(message)
        llm_result = self._llm.post_chat_request(context, messages)

        if len(llm_result.choices) < 1:
            return self._parse_line("")
        response = llm_result.first_choice.lower()
        parsed = [self._parse_line(line) for line in response.split("\n") if line.strip().startswith("score")]

        return parsed[0]

    def _build_messages(self, message: ChatMessage) -> List[LLMMessage]:
        user_prompt = [
            "Please give the similarity score of the actual message compared to the expected one.",
            "Actual message:",
            message.message,
            "Expected message:",
            self._expected,
        ]

        result = [self._system_message, LLMMessage.user_message("\n".join(user_prompt))]
        return result

    @staticmethod
    def _build_system_prompt() -> LLMMessage:
        system_prompt = [
            "# Role:",
            "You are an assistant specialized in evaluating how similar an expected message and an actual message are.",
            "# Instructions:",
            "Compare the {expected} message and the {actual} message",
            "Give a similarity score out of 100%",
            "Unrelated messages have a 0% similarity score",
            "Provide the result exactly in the format `score: {similarity score} - short justification`",
        ]
        return LLMMessage.system_message("\n".join(system_prompt))

    @staticmethod
    def _parse_line(line: str) -> float:
        line = line.lower().removeprefix("score").strip().replace("-", ":")
        try:
            score = line.split(":", 3)
            return float(score[1].strip(":% ")) / 100.0
        except Exception:
            raise
