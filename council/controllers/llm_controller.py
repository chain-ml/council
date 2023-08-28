import logging
from typing import List, Tuple

from council.contexts import AgentContext, ScoredChatMessage
from council.chains import Chain
from council.llm import LLMMessage, LLMBase
from council.utils import Option
from council.runners import Budget

from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit
from ..monitors import Monitored

logger = logging.getLogger(__name__)


class LLMController(ControllerBase):
    """
    A controller that uses an LLM to decide the execution plan
    """

    _llm: LLMBase
    _monitored_llm: Monitored[LLMBase]

    def __init__(self, llm: LLMBase, response_threshold: float = 0, top_k_execution_plan: int = 10000):
        """
        Initialize a new instance

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
            top_k_execution_plan (int): maximum number of execution plan returned
        """
        super().__init__()
        self._llm = llm
        self._monitored_llm = self.new_monitor("llm", self._llm)
        self._response_threshold = response_threshold
        self._top_k = top_k_execution_plan

    def select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        return context.evaluationHistory[-1]

    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        messages = self._build_llm_messages(chains, context)
        with context.new_for(self._monitored_llm).new_log_entry() as log_entry:
            llm_result = self._llm.monitored_post_chat_request(log_entry, messages)
        response = llm_result.first_choice
        logger.debug(f"llm response: {response}")

        parsed = [self._parse_line(line, chains) for line in response.splitlines() if line.lower().startswith("name:")]
        filtered = [r.unwrap() for r in parsed if r.is_some() and r.unwrap()[1] > self._response_threshold]
        if (filtered is None) or (len(filtered) == 0):
            return []

        filtered.sort(key=lambda item: item[1], reverse=True)
        result = [ExecutionUnit(r[0], budget) for r in filtered if r is not None]

        return result[: self._top_k]

    @staticmethod
    def _build_llm_messages(chains, context):
        answer_choices = "\n ".join([f"name: {c.name}, description: {c.description}" for c in chains])
        task_description = [
            "# Role:",
            "You are an assistant responsible to identify the intent of the user against a list of categories.",
            "Categories are given as a name and a description formatted precisely as:",
            "name: {name}, description: {description})",
            answer_choices,
            "# Instructions:",
            "# Score how relevant a category is from 0 to 10 using their description",
            "# For each category, your scores will be formatted precisely as:",
            "Name: {name};Score: {score as int};{short justification}",
            "# When no category is relevant, you will answer exactly with 'unknown'",
        ]
        messages = [
            LLMMessage.system_message("\n".join(task_description)),
            LLMMessage.user_message(
                f"What are most relevant categories for:\n {context.chatHistory.try_last_user_message.unwrap().message}"
            ),
        ]
        return messages

    @staticmethod
    def _parse_line(line: str, chains: List[Chain]) -> Option[Tuple[Chain, int]]:
        result: Option[Tuple[Chain, int]] = Option.none()
        name: str = ""
        score: str = ""
        line = line.lower().removeprefix("name:")
        try:
            (name, score, _j) = line.split(";", 3)
            name = name.strip().casefold()
            chain = next(filter(lambda item: item.name.casefold() == name, chains))
            score = score.replace("score:", "").strip()
            result = Option.some((chain, int(score)))
        except StopIteration:
            logger.warning(f'message="no chain found with name `{name}`"')
        finally:
            return result
