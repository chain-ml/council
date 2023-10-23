"""
LLMEvaluator implementation.

This evaluator uses the given `LLM` to evaluate the chain's responses.
"""
from typing import List, Optional

from council.contexts import AgentContext, ChatMessage, ContextLogger, ScoredChatMessage
from council.evaluators import EvaluatorBase
from council.llm import LLMBase, LLMMessage, MonitoredLLM, llm_property, LLMAnswer
from council.utils import Option


class SpecialistGrade:
    def __init__(self, index: int, grade: float, justification: str):
        self._grade = grade
        self._index = index
        self._justification = justification

    @llm_property
    def grade(self) -> float:
        """Your Grade"""
        return self._grade

    @llm_property
    def index(self) -> int:
        """Index of the answer graded"""
        return self._index

    @llm_property
    def justification(self):
        """Short and specific explanation of this particular grade"""
        return self._justification


class LLMEvaluator(EvaluatorBase):
    """Evaluator using an `LLM` to evaluate chain responses."""

    def __init__(self, llm: LLMBase):
        """
        Build a new LLMEvaluator.

        :param llm: model to use for the evaluation.
        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._llm_evaluator_answer = LLMAnswer(SpecialistGrade)
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        query = context.chat_history.try_last_user_message.unwrap()
        chain_results = [
            chain_messages.try_last_message.unwrap()
            for chain_messages in context.chains
            if chain_messages.try_last_message.is_some()
        ]

        retry = self._retry
        messages = self._build_llm_messages(query, chain_results)
        while retry > 0:
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                parse_response = self._parse_response(response, chain_results)
                return parse_response
            except Exception as e:
                messages.append(LLMMessage.assistant_message("Your response raised an exception:\n" + response))
                messages.append(LLMMessage.user_message(f"{e.__class__.__name__}: `{e}`"))
                retry -= 1
        context.logger.debug(f"TODO")
        return []

    def _parse_response(self, response: str, chain_results: List[ChatMessage]) -> List[ScoredChatMessage]:
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        grades = [r.unwrap() for r in parsed if r.is_some()]

        scored_messages = []
        for idx, message in enumerate(chain_results):
            try:
                grade = next(filter(lambda item: item.index == (idx + 1), grades))
                scored_message = ScoredChatMessage(
                    ChatMessage.agent(message=message.message, data=message.data), grade.grade
                )
                scored_messages.append(scored_message)
            except StopIteration:
                raise Exception(f"Please grade ALL {len(chain_results)} answers.")

        return scored_messages

    def _build_llm_messages(self, query: ChatMessage, skill_messages: list[ChatMessage]) -> List[LLMMessage]:
        if len(skill_messages) <= 0:
            return []

        responses = [skill_message.message for skill_message in skill_messages]
        prompt = self._build_system_prompt_multiple_answers()
        return [
            LLMMessage.system_message(prompt),
            LLMMessage.user_message(self._build_multiple_answers_message(query.message, responses)),
        ]

    def _parse_line(self, line: str) -> Option[SpecialistGrade]:
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[SpecialistGrade] = self._llm_evaluator_answer.to_object(line)
        if cs is not None:
            return Option.some(cs)
        return Option.none()

    @staticmethod
    def _build_multiple_answers_message(query: str, answers: list[str]) -> str:
        prompt_answers = "\n".join(
            f"- answer #{index + 1} is: {answer if len(answer) > 0 else 'EMPTY'}"
            for index, answer in enumerate(answers)
        )
        lines = [
            "The question to grade is:",
            query,
            "Please grade the following answers according to your instructions:",
            prompt_answers,
        ]
        return "\n".join(lines)

    def _build_system_prompt_multiple_answers(self) -> str:
        """Build prompt that will be sent to the inner `LLM`."""
        task_description = [
            "# ROLE",
            "You are an expert grading fairly answers from different Specialists to a given question.",
            "",
            "# INSTRUCTIONS",
            "1. Give a grade from 0.0 to 10.0",
            "2. Evaluate carefully step by step the question and the proposed answer before grading.",
            "3. Grade independently of the order of an answer.",
            "4. Ensure to be consistent in grading, identical answers must have the same grade.",
            "5. Irrelevant, inaccurate, inappropriate, false or empty answer must be graded 0.0",
            "6. Math problem should be accurate.",
            "",
            "# FORMATTING",
            "1. The list of given answers is formatted precisely as:",
            "- answer #{index} is: {answer or EMPTY if no answer}",
            "2. For each given answer, format your response precisely as:",
            self._llm_evaluator_answer.to_prompt(),
        ]
        prompt = "\n".join(task_description)
        return prompt
