from typing import List, Optional, Sequence, Tuple

from council.chains import ChainBase
from council.contexts import AgentContext, ChatMessage
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.utils import Option
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit


class LLMController(ControllerBase):
    """
    A controller that uses an LLM to decide the execution plan
    """

    _llm: MonitoredLLM

    def __init__(
        self,
        chains: Sequence[ChainBase],
        llm: LLMBase,
        response_threshold: float = 0.0,
        top_k: Optional[int] = None,
        parallelism: bool = False,
    ):
        """
        Initialize a new instance of an LLMController

        Parameters:
            llm (LLMBase): the instance of LLM to use
            response_threshold (float): a minimum threshold to select a response from its score
            top_k (int): maximum number of execution plan returned
            parallelism (bool): If true, Build a plan that will be executed in parallel
        """
        super().__init__(chains=chains, parallelism=parallelism)
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
        parsed = [self._parse_line(context, line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in filtered][: self._top_k]

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
                f"Score categories for:\n {context.chat_history.try_last_user_message.unwrap().message}"
            ),
        ]
        return messages

    def _build_system_message(self) -> LLMMessage:
        answer_choices = "\n".join(
            [f"name: {c.name};description: {c.description};{c.is_supporting_instructions}" for c in self._chains]
        )

        if self._top_k == 1:
            instruction = "Score only the most relevant category. Your answer will be formatted precisely as:"
        else:
            instruction = (
                f"Score all {len(self._chains)} given categories. All your answers will be formatted precisely as:"
            )
        task_description = [
            "\n# ROLE:",
            "You are an assistant responsible to identify the intent of the user against a list of categories.",
            "Categories are given as a name and a description formatted precisely as:",
            "name: {name}, description: {description}, boolean indicating if supporting instructions",
            "\n# CATEGORIES:",
            answer_choices,
            "\n# INSTRUCTIONS:",
            "Score how relevant a category is from 0 to 10 using their description.",
            instruction,
            "Name: {category name}<->"
            "Score: {your score as int}<->"
            "Instructions: {your instructions IF category supports instruction ELSE none}<->"
            "Justification: {short justification}",
        ]
        return LLMMessage.system_message("\n".join(task_description))

    def _parse_line(self, context: AgentContext, line: str) -> Option[Tuple[ExecutionUnit, int]]:
        line = line.lower()
        if not line.startswith("name:"):
            return Option.none()
        else:
            line = line.removeprefix("name:")

        maybe_name: str = ""
        maybe_score: str = ""
        try:
            (maybe_name, maybe_score, instructions, _j) = line.split("<->", 4)
            name = maybe_name.strip().casefold()
            chain = next(filter(lambda item: item.name.casefold() == name, self._chains))

            maybe_score = maybe_score.replace("score:", "").strip()
            score = int(maybe_score)
            if score < self._response_threshold:
                return Option.none()

            instructions = instructions.replace("instructions:", "")
            return Option.some((self._build_execution_unit(chain, context, instructions, score), score))

        except StopIteration:
            context.logger.warning(f'message="no chain found with name `{maybe_name}`"')
        except ValueError:
            context.logger.warning(f'message="invalid score `{maybe_score}`"')
        return Option.none()

    def _build_execution_unit(self, chain: ChainBase, context: AgentContext, instructions: str, score: int):
        return ExecutionUnit(
            chain,
            context.budget,
            initial_state=ChatMessage.chain(message=instructions) if chain.is_supporting_instructions else None,
            name=f"{chain.name};{score}",
            rank=self.default_execution_unit_rank,
        )
