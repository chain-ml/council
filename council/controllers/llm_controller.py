import logging
from typing import List, Optional, Sequence, Tuple

from council.chains import ChainBase
from council.contexts import AgentContext
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.utils import Option
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit

logger = logging.getLogger(__name__)


class LLMController(ControllerBase):
    """
    A controller that uses an LLM to decide the execution plan
    """

    _llm: MonitoredLLM

    def __init__(
        self, chains: Sequence[ChainBase], llm: LLMBase, response_threshold: float = 0.0, top_k: Optional[int] = None
    ):
        """
        Initialize a new instance of an LLMController

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
            top_k (int): maximum number of execution plan returned
        """
        super().__init__(chains=chains)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._response_threshold = response_threshold
        self._top_k = top_k
        self._llm_system_message = self._build_system_message()

    @property
    def llm(self) -> LLMBase:
        """
        the LLM used by the controller
        """
        return self._llm.inner

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        response = self._call_llm(context)
        parsed = [
            self._parse_line(line, self._chains)
            for line in response.strip().splitlines()
            if line.lower().startswith("name:")
        ]

        filtered = [r.unwrap() for r in parsed if r.is_some() and r.unwrap()[1] > self._response_threshold]
        if (filtered is None) or (len(filtered) == 0):
            return []

        filtered.sort(key=lambda item: item[1], reverse=True)
        result = [
            ExecutionUnit(chain, context.budget, name=f"{chain.name};{score}")
            for chain, score in filtered
            if chain is not None
        ]

        if self._top_k is not None and self._top_k > 0:
            return result[: self._top_k]
        return result

    def _call_llm(self, context: AgentContext) -> str:
        messages = self._build_llm_messages(context)
        llm_result = self._llm.post_chat_request(context, messages)
        response = llm_result.first_choice
        context.logger.debug(f"llm response: {response}")
        return response

    def _build_llm_messages(self, context: AgentContext) -> List[LLMMessage]:
        messages = [
            self._llm_system_message,
            LLMMessage.user_message(
                "What are most relevant categories"
                f"for:\n {context.chat_history.try_last_user_message.unwrap().message}"
            ),
        ]
        return messages

    def _build_system_message(self) -> LLMMessage:
        answer_choices = "\n ".join([f"name: {c.name}, description: {c.description}" for c in self._chains])
        task_description = [
            "# Role:",
            "You are an assistant responsible to identify the intent of the user against a list of categories.",
            "Categories are given as a name and a description formatted precisely as:",
            "name: {name}, description: {description})",
            answer_choices,
            "# INSTRUCTIONS:",
            "# Score how relevant a category is from 0 to 10 using their description",
            "# For each category, your scores will be formatted precisely as:",
            "Name: {name};Score: {score as int};{short justification}",
            "# When no category is relevant, you will answer exactly with 'unknown'",
        ]
        return LLMMessage.system_message("\n".join(task_description))

    @staticmethod
    def _parse_line(line: str, chains: Sequence[ChainBase]) -> Option[Tuple[ChainBase, int]]:
        result: Option[Tuple[ChainBase, int]] = Option.none()
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
