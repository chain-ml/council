"""
LLMEvaluator implementation.

This evaluator uses the given `LLM` to evaluate the chain's responses.
"""
import logging
from typing import List

from council.contexts import AgentContext, ScoredChatMessage, ChatMessage
from council.evaluators import EvaluatorBase
from council.llm import LLMBase, LLMMessage
from council.monitors import Monitored
from council.runners import Budget
from council.utils import Option


class LLMEvaluator(EvaluatorBase):
    """Evaluator using an `LLM` to evaluate chain responses."""

    def __init__(self, llm: LLMBase):
        """
        Build a new LLMEvaluator.

        :param llm: model to use for the evaluation.
        """
        super().__init__()
        self._llm = llm
        self._monitored_llm = Monitored("llm", self._llm)

    def execute(self, context: AgentContext, budget: Budget) -> List[ScoredChatMessage]:
        query = context.chatHistory.try_last_user_message.unwrap()
        chain_results = [
            chain_messages.try_last_message.unwrap()
            for chain_messages in context.chains
            if chain_messages.try_last_message.is_some()
        ]
        scored_messages = self.__score_responses(
            context=context, query=query, skill_messages=chain_results, budget=budget
        )
        return list(scored_messages)

    def __score_responses(
        self, context: AgentContext, query: ChatMessage, skill_messages: list[ChatMessage], budget: Budget
    ) -> List[ScoredChatMessage]:
        """
        Score agent response.

        :param query: Query used to build the responses.
        :param skill_messages: Responses generated by the chain.
        :return: list of scored messages.
        """
        # Build prompt to send to the inner LLM
        if len(skill_messages) <= 0:
            return []
        elif len(skill_messages) == 1:
            prompt = self._build_system_prompt_single_answer()
            messages = [
                LLMMessage.system_message(prompt),
                LLMMessage.user_message(self._build_single_answer_message(query.message, skill_messages[0].message)),
            ]
        else:
            responses = [skill_message.message for skill_message in skill_messages]
            prompt = self._build_system_prompt_multiple_answers()
            messages = [
                LLMMessage.system_message(prompt),
                LLMMessage.user_message(self._build_multiple_answers_message(query.message, responses)),
            ]

        with context.new_agent_context_for(self._monitored_llm).log_entry as log_entry:
            result = self._llm.monitored_post_chat_request(log_entry=log_entry, messages=messages)
        for c in result.consumptions:
            budget.add_consumption(c, "LLMEvaluator")
        llm_response = result.first_choice

        # Parse LLM response with the score for each message we want to score
        scores = [self._parse_eval(line) for line in llm_response.split("\n") if line.lower().startswith("grade")]

        agent_messages = []
        for skill_message, score in filter(lambda tuple: tuple[1].is_some(), zip(skill_messages, scores)):
            agent_message = ScoredChatMessage(
                ChatMessage.agent(message=skill_message.message, data=skill_message.data), score.unwrap()
            )
            agent_messages.append(agent_message)

        return agent_messages

    @staticmethod
    def _parse_eval(line: str) -> Option[float]:
        """Parse the evaluation response from the inner `LLM`."""

        line = line.lower().removeprefix("answer").strip().replace("-", ":")
        try:
            score = line.split(":", 3)
            return Option.some(float(score[1]))
        except ValueError:
            logging.exception(f'message="could not parse score" line="{line}"')
            raise
        except Exception:
            logging.exception(f'message="could not parse evaluation response" line="{line}"')
            raise

    @staticmethod
    def _build_multiple_answers_message(query: str, answers: list[str]) -> str:
        prompt_answers = "\n".join(f"Answer #{index+1} is:\n{answer}" for index, answer in enumerate(answers))
        lines = ["# The question to grade is:", query, "# The given answers are:", prompt_answers, "# Please grade."]
        return "\n".join(lines)

    @staticmethod
    def _build_single_answer_message(query: str, answer: str) -> str:
        lines = ["# The question to grade is:", query, "# The given answer is:", answer, "# Please grade."]
        return "\n".join(lines)

    @staticmethod
    def _build_system_prompt_multiple_answers() -> str:
        """Build prompt that will be sent to the inner `LLM`."""
        task_description = [
            "# You are a grading expert, grading how accurate and relevant multiple answers are to a given question.",
            "# Your grade will only be based on the given answer.",
            "# The list of given answers is formatted precisely as:",
            "Answer #{index} is:",
            "{answer}",
            "# INSTRUCTIONS: ",
            "# Give a grade from 0.0 to 10.0",
            "# Same answers must have the same grade.",
            "# Irrelevant or empty answer must be graded 0.0",
            "# For each given answer, your grade will be formatted precisely as:",
            "grade #{index}: {grade as float} - short justification",
        ]
        prompt = "\n".join(task_description)
        return prompt

    @staticmethod
    def _build_system_prompt_single_answer() -> str:
        """Build prompt that will be sent to the inner `LLM`."""

        task_description = [
            "# You are a grading expert, grading how accurate and relevant an answer is to a given question.",
            "# INSTRUCTIONS: ",
            "# Give a grade from 0.0 to 10.0",
            "# Irrelevant or empty answer must be graded 0.0",
            "# Your grade will be formatted precisely as:",
            "grade: {grade as float} - short justification",
        ]
        prompt = "\n".join(task_description)
        return prompt
