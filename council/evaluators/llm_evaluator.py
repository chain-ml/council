"""
LLMEvaluator implementation.

This evaluator uses the given `LLM` to evaluate the chain's responses.
"""
from typing import List, Optional

from council.contexts import AgentContext, ChatMessage, ScoredChatMessage, ContextBase
from council.evaluators import EvaluatorBase, EvaluatorException
from council.llm import LLMBase, MonitoredLLM, llm_property, LLMAnswer, LLMMessage
from council.llm.llm_answer import LLMParsingException
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
        """Index of the answer graded in the list"""
        return self._index

    @llm_property
    def justification(self) -> str:
        """Short, helpful and specific explanation your grade"""
        return self._justification

    def __str__(self):
        return f"Message `{self._index}` graded `{self._grade}` with the justification: `{self._justification}`"


class LLMEvaluator(EvaluatorBase):
    """Evaluator using an `LLM` to evaluate chain responses."""

    def __init__(self, llm: LLMBase):
        """
        Build a new LLMEvaluator.

        :param llm: model to use for the evaluation.
        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._llm_answer = LLMAnswer(SpecialistGrade)
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
        new_messages: List[LLMMessage] = []
        while retry > 0:
            retry -= 1
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                parse_response = self._parse_response(context, response, chain_results)
                return parse_response
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except EvaluatorException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise EvaluatorException("LLMEvaluator failed to execute.")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _parse_response(
        self, context: ContextBase, response: str, chain_results: List[ChatMessage]
    ) -> List[ScoredChatMessage]:
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        grades = [r.unwrap() for r in parsed if r.is_some()]
        if len(grades) == 0:
            raise LLMParsingException("None of your grade could be parsed. Follow exactly formatting instructions.")

        scored_messages = []
        missing_indexes = []
        for idx, message in enumerate(chain_results):
            try:
                grade = next(filter(lambda item: item.index == (idx + 1), grades))
                scored_message = ScoredChatMessage(
                    ChatMessage.agent(message=message.message, data=message.data), grade.grade
                )
                scored_messages.append(scored_message)
                context.logger.debug(f"{grade} Graded message: `{message.message}`")
            except StopIteration:
                missing_indexes.append(idx + 1)

        if len(missing_indexes) > 1:
            missing_msg = f"Missing grade for the answers with indexes {missing_indexes}."
            raise EvaluatorException(f"Grade ALL {len(chain_results)} answers. {missing_msg}")

        if len(missing_indexes) > 0:
            missing_msg = f"Missing grade for the answer with index {missing_indexes[0]}."
            raise EvaluatorException(f"Grade ALL {len(chain_results)} answers. {missing_msg}")

        return scored_messages

    def _build_llm_messages(self, query: ChatMessage, skill_messages: List[ChatMessage]) -> List[LLMMessage]:
        if len(skill_messages) <= 0:
            return []

        responses = [skill_message.message for skill_message in skill_messages]
        return [self._build_system_message(), self._build_user_message(query.message, responses)]

    def _parse_line(self, line: str) -> Option[SpecialistGrade]:
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[SpecialistGrade] = self._llm_answer.to_object(line)
        return Option(cs)

    @staticmethod
    def _build_user_message(query: str, answers: list[str]) -> LLMMessage:
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
        prompt = "\n".join(lines)
        return LLMMessage.user_message(prompt)

    def _build_system_message(self) -> LLMMessage:
        """Build prompt that will be sent to the inner `LLM`."""
        task_description = [
            "\n# ROLE",
            "You are an instructor, with a large breadth of knowledge.",
            "You are grading with objectivity answers from different Specialists to a given question.",
            "\n# INSTRUCTIONS",
            "1. Give a grade from 0.0 to 10.0",
            "2. Evaluate carefully the question and the proposed answer.",
            "3. Ignore how assertive the answer is, only content accuracy count for grading."
            "4. Consider only the Specialist's answer and ignore its index for grading.",
            "5. Ensure to be consistent in grading, identical answers must have the same grade.",
            "6. Irrelevant, inaccurate, inappropriate, false or empty answer must be graded 0.0",
            "\n# FORMATTING",
            "1. The list of given answers is formatted precisely as:",
            "- answer #{index} is: {Specialist's answer or EMPTY if no answer}",
            "2. For each given answer, format your response precisely as:",
            self._llm_answer.to_prompt(),
        ]
        prompt = "\n".join(task_description)
        return LLMMessage.system_message(prompt)
