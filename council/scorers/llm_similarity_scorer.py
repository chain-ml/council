from typing import List, Dict, Any, Optional

from . import ScorerException
from .scorer_base import ScorerBase
from council.contexts import ChatMessage, ScorerContext, ContextBase
from council.llm import LLMBase, LLMMessage, MonitoredLLM, llm_property, LLMAnswer
from ..llm.llm_answer import LLMParsingException
from ..utils import Option


class SimilarityScore:
    def __init__(self, score: float, justification: str):
        self._score = score
        self._justification = justification

    @llm_property
    def score(self) -> float:
        """Your similarity Score"""
        return self._score / 100.0

    @llm_property
    def justification(self) -> str:
        """Short, helpful and specific explanation your score"""
        return self._justification

    def __str__(self):
        return f"Similarity score is {self.score} with the justification: {self._justification}"


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
        self._llm_answer = LLMAnswer(SimilarityScore)
        self._system_message = self._build_system_message()
        self._retry = 3

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["expected"] = self._expected
        return result

    def _score(self, context: ScorerContext, message: ChatMessage) -> float:
        retry = self._retry
        messages = self._build_llm_messages(message)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                retry -= 1
                similarity_score = self._parse_response(context, response)
                return similarity_score.score
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except ScorerException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise ScorerException("LLMSimilarityScorer failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _build_llm_messages(self, message: ChatMessage) -> List[LLMMessage]:
        user_prompt = [
            "Please give the similarity score of the actual message compared to the expected one.",
            "Actual message:",
            message.message,
            "Expected message:",
            self._expected,
        ]

        result = [self._system_message, LLMMessage.user_message("\n".join(user_prompt))]
        return result

    def _build_system_message(self) -> LLMMessage:
        system_prompt = [
            "# ROLE",
            "You are an expert specialized in evaluating how similar an expected message and an actual message are.",
            "\n# INSTRUCTIONS",
            "1. Compare the {expected} message and the {actual} message.",
            "2. Score 0 (2 messages are unrelated) to 100 (the 2 messages have the same content).",
            "3. Your score must be fair.",
            "\n#FORMATTING",
            self._llm_answer.to_prompt(),
        ]
        return LLMMessage.system_message("\n".join(system_prompt))

    def _parse_response(self, context: ContextBase, response: str) -> SimilarityScore:
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        if len(filtered) == 0:
            raise LLMParsingException("None of your response could be parsed. Follow exactly formatting instructions.")

        similarity_score = filtered[0]
        context.logger.debug(f"{similarity_score}")
        return similarity_score

    def _parse_line(self, line: str) -> Option[SimilarityScore]:
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        similarity_score: Optional[SimilarityScore] = self._llm_answer.to_object(line)
        if similarity_score is not None:
            return Option.some(similarity_score)
        return Option.none()
