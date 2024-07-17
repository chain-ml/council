"""
LLMFilter implementation.

This filter uses the given `LLM` to filter the chain's responses.
"""

from typing import List, Optional

from council.contexts import AgentContext, ContextBase, ScoredChatMessage
from council.filters import FilterBase, FilterException
from council.llm import LLMAnswer, LLMBase, LLMMessage, MonitoredLLM, llm_property
from council.llm.llm_answer import LLMParsingException
from council.utils import Option


class FilterResult:
    def __init__(self, index: int, is_filtered: bool, justification: str) -> None:
        self._filtered = is_filtered
        self._index = index
        self._justification = justification

    @llm_property
    def is_filtered(self) -> bool:
        """Filter response"""
        return self._filtered

    @llm_property
    def index(self) -> int:
        """Index of the answer in the list"""
        return self._index

    @llm_property
    def justification(self) -> str:
        """Short, helpful and specific explanation your response"""
        return self._justification

    def __str__(self) -> str:
        t = " " if self._filtered else " not "
        return f"Message {self._index} is{t}filtered with the justification: {self._justification}"


class LLMFilter(FilterBase):
    """Filter using an `LLM` to filter chain responses."""

    def __init__(self, llm: LLMBase, filter_on: Optional[List[str]] = None) -> None:
        """
        Build a new LLMFilter.

        :param llm: model to use for the filtering.
        :param filter_on: List of filters to filter chain responses on.
        """
        super().__init__()
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._llm_answer = LLMAnswer(FilterResult)
        self._filter_on = filter_on or []
        self._retry = 3

    def _execute(self, context: AgentContext) -> List[ScoredChatMessage]:
        all_eval_results = list(context.evaluation)
        if all_eval_results is None:
            return []

        if len(self._filter_on) == 0:
            return all_eval_results

        retry = self._retry
        messages = self._build_llm_messages(all_eval_results)
        new_messages: List[LLMMessage] = []
        while retry > 0:
            retry -= 1
            messages = messages + new_messages
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                return self._parse_response(context, response, all_eval_results)
            except LLMParsingException as e:
                assistant_message = f"Your response is not correctly formatted:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)
            except FilterException as e:
                assistant_message = f"Your response raised an exception:\n{response}"
                new_messages = self._handle_error(e, assistant_message, context)

        raise FilterException(f"LLMFilter failed to execute after {self._retry} retries")

    @staticmethod
    def _handle_error(e: Exception, assistant_message: str, context: ContextBase) -> List[LLMMessage]:
        error = f"{e.__class__.__name__}: `{e}`"
        context.logger.warning(f"Exception occurred: {error}")
        return [LLMMessage.assistant_message(assistant_message), LLMMessage.user_message(f"Fix:\n{error}")]

    def _parse_response(
        self, context: ContextBase, response: str, messages: List[ScoredChatMessage]
    ) -> List[ScoredChatMessage]:
        parsed = [self._parse_line(line) for line in response.strip().splitlines()]
        answers = [r.unwrap() for r in parsed if r.is_some()]
        if len(answers) == 0:
            raise LLMParsingException("None of your answer could be parsed. Follow exactly formatting instructions.")

        messages_to_keep = []
        missing = []
        for idx, message in enumerate(messages):
            try:
                answer = next(filter(lambda item: item.index == (idx + 1), answers))
                if not answer.is_filtered:
                    messages_to_keep.append(message)
                context.logger.debug(f"{answer} for {message.message}")
            except StopIteration:
                missing.append(idx)

        if len(missing) > 0:
            raise FilterException(
                f"Please evaluate ALL {len(messages)} answers. Missing filter responses for {missing} answers."
            )

        return messages_to_keep

    def _build_llm_messages(self, messages: List[ScoredChatMessage]) -> List[LLMMessage]:
        return [self._build_system_message(), self._build_user_message(messages)]

    def _parse_line(self, line: str) -> Option[FilterResult]:
        if LLMAnswer.field_separator() not in line:
            return Option.none()

        cs: Optional[FilterResult] = self._llm_answer.to_object(line)
        return Option(cs)

    def _build_user_message(self, messages: List[ScoredChatMessage]) -> LLMMessage:
        prompt_answers = "\n".join(
            f"- answer #{index + 1} is: {message.message}" for index, message in enumerate(messages)
        )
        filters = "\n".join(f"- {filter}" for filter in self._filter_on)

        lines = [
            "\nFILTERS",
            filters,
            "\nPlease filter or not the following answers according to your instructions:",
            prompt_answers,
        ]
        prompt = "\n".join(lines)
        return LLMMessage.user_message(prompt)

    def _build_system_message(self) -> LLMMessage:
        """Build prompt that will be sent to the inner `LLM`."""
        task_description = [
            "\n# ROLE",
            "You are a judge, with a large breadth of knowledge.",
            "You are deciding with objectivity if some answers from different Specialists need to be filtered.",
            "\n# INSTRUCTIONS",
            "1. Give your response with TRUE or FALSE",
            "2. Evaluate carefully and fairly the proposed answer.",
            "3. Ignore how assertive the answer is, only content accuracy count."
            "4. Consider only the Specialist's answer and ignore its index.",
            "5. Ensure to be consistent, identical answers must have the same response.",
            "\n# FORMATTING",
            "1. The list of given answers is formatted precisely as:",
            "- answer #{index} is: {Specialist's answer or EMPTY if no answer}",
            "2. For each given answer, format your response precisely as:",
            self._llm_answer.to_prompt(),
        ]
        prompt = "\n".join(task_description)
        return LLMMessage.system_message(prompt)
