import logging
from typing import List, Tuple

from council.contexts import AgentContext, ScoredChatMessage
from council.chains import Chain
from council.llm import LLMMessage, LLMBase
from council.utils import Option
from council.runners import Budget

from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit

logger = logging.getLogger(__name__)


class LLMController(ControllerBase):
    """
    A controller that uses an LLM to decide the execution plan
    """

    _llm: LLMBase

    def __init__(self, llm: LLMBase, response_threshold: float = 0, top_k_execution_plan: int = 10000):
        """
        Initialize a new instance

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
            top_k_execution_plan (int): maximum number of execution plan returned
        """
        self._llm = llm
        self._response_threshold = response_threshold
        self._top_k = top_k_execution_plan

    def select_responses(self, context: AgentContext) -> List[ScoredChatMessage]:
        return context.evaluationHistory[-1]

    def get_plan(self, context: AgentContext, chains: List[Chain], budget: Budget) -> List[ExecutionUnit]:
        answer_choices = "\n ".join([f"name: {c.name}, description: {c.description}" for c in chains])
        task_description = [
            "You are an assistant responsible to identify the intent of the user. ",
            "Categories are given as a name and a category (name: {name}, description: {description})",
            answer_choices,
            "Instructions:" "# score categories out of 10 using there description",
            "# For each category, you will answer with {name};{score}",
            "# Each response is provided on a new line",
            "# When no category is relevant, you will answer exactly with 'unknown'",
        ]

        messages = [
            LLMMessage.system_message("\n".join(task_description)),
            LLMMessage.user_message(
                f"what are most relevant categories for: {context.chatHistory.try_last_user_message.unwrap().message}"
            ),
        ]

        response = self._llm.post_chat_request(messages)[0]
        logger.debug(f"llm response: {response}")

        parsed = [self.parse_line(line, chains) for line in response.splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some() and r.unwrap()[1] > self._response_threshold]
        if (filtered is None) or (len(filtered) == 0):
            return []

        filtered.sort(key=lambda item: item[1], reverse=True)
        result = [ExecutionUnit(r[0], budget) for r in filtered if r is not None]

        return result[: self._top_k]

    @staticmethod
    def parse_line(line: str, chains: List[Chain]) -> Option[Tuple[Chain, int]]:
        result: Option[Tuple[Chain, int]] = Option.none()
        name = ""
        try:
            (name, score) = line.split(";", 2)
            name = name.casefold()
            chain = next(filter(lambda item: item.name.casefold() == name, chains))
            result = Option.some((chain, int(score)))
        except StopIteration:
            logger.warning(f'message="not chain found with name `{name}`"')
        finally:
            return result
