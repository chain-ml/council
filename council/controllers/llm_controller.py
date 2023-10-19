from typing import List, Optional, Sequence, Tuple

from council.chains import ChainBase
from council.contexts import AgentContext, ChatMessage
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.utils import Option
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit
from .llm_answer import llm_property, LLMControllerAnswer


class Specialist:
    def __init__(self, name: str, justification: str, instructions: str, score: int):
        self._instructions = instructions
        self._score = score
        self._name = name
        self._justification = justification

    @llm_property
    def name(self) -> str:
        """Specialist name to score for the task"""
        return self._name

    @llm_property
    def score(self) -> int:
        """Specialist relevance score from 0 to 10. 0 if poor relevance or out of scope, and 10 if perfectly relevant"""
        return self._score

    @llm_property
    def instructions(self) -> str:
        """Specific instructions to this particular specialist, or None if the specialist is not supporting those."""
        return self._instructions

    @llm_property
    def justification(self):
        """Short and specific explanation of this particular specialist score"""
        return self._justification


class LLMPrompter:
    def __init__(self, llm: LLMBase, system_message: str, prompt: str):
        self._llm = llm
        self._system_message = system_message
        self._prompt = prompt


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
        if top_k is None:
            self._top_k = len(self._chains)
        else:
            self._top_k = min(top_k, len(self._chains))
        self._llm_controller_answer = LLMControllerAnswer(Specialist)
        self._llm_system_message = self._build_system_message()

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        retry = 2
        messages = self._build_llm_messages(context)
        while retry > 0:
            llm_result = self._llm.post_chat_request(context, messages)
            response = llm_result.first_choice
            context.logger.debug(f"llm response: {response}")
            try:
                plan = self._parse_response(context, response)
                plan.sort(key=lambda item: item[1], reverse=True)
                return [item[0] for item in plan if item[1] >= self._response_threshold][: self._top_k]
            except Exception as e:
                messages.append(LLMMessage.assistant_message(response))
                messages.append(LLMMessage.user_message(f"{e.__class__.__name__}: {e}"))
                retry -= 1

    def _parse_response(self, context: AgentContext, response: str) -> List[Tuple[ExecutionUnit, int]]:
        parsed = [self._parse_line(context, line) for line in response.strip().splitlines()]
        filtered = [r.unwrap() for r in parsed if r.is_some()]
        if self._top_k > 1:
            actual_chains = [item[0].chain.name for item in filtered]
            missing_chains = [chain.name for chain in self._chains if chain.name not in actual_chains]
            if len(missing_chains) > 0:
                raise Exception(f"Missing scores for {missing_chains}.")

        if len(filtered) != 1 and self._top_k == 1:
            raise Exception(f"You score all Specialists. ONLY score only the most relevant specialist.")

        return filtered

    def _build_llm_messages(self, context: AgentContext) -> List[LLMMessage]:
        messages = [
            self._llm_system_message,
        ]
        message = context.chat_history.try_last_user_message
        if message.is_some():
            messages.append(LLMMessage.user_message(f"Score Specialists for:\n `{message.unwrap().message}`"))
        return messages

    def _build_system_message(self) -> LLMMessage:
        answer_choices = "\n".join(
            [f"name: {c.name};description: {c.description};{c.is_supporting_instructions}" for c in self._chains]
        )

        if self._top_k == 1:
            instruction = "- Score only the most relevant and best Specialist."
        else:
            instruction = "- Score all Specialists."
        task_description = [
            "\n# INSTRUCTIONS",
            "You are an assistant responsible to carefully select Specialist by giving them a relevance score.",
            "- Use the Specialist description ONLY to score its relevance, ignore its name.",
            instruction,
            "- Specialist information are precisely formatted as:",
            "`name: {name};description: {description};{boolean indicating if Specialists is supporting instructions}`",
            "\n# SPECIALISTS",
            answer_choices,
            self._llm_controller_answer.to_prompt(),
        ]
        return LLMMessage.system_message("\n".join(task_description))

    def _parse_line(self, context: AgentContext, line: str) -> Option[Tuple[ExecutionUnit, int]]:
        if LLMControllerAnswer.field_separator() not in line:
            return Option.none()
        cs: Specialist = self._llm_controller_answer.to_object(line)
        if cs is not None:
            try:
                chain = next(filter(lambda item: item.name.casefold() == cs.name.casefold(), self._chains))
                return Option.some((self._build_execution_unit(chain, context, cs.instructions, cs.score), cs.score))
            except StopIteration:
                context.logger.warning(f'message="no chain found with name `{cs.name}`"')
                raise Exception(f"The Specialist `{cs.name}` does not exist")
        return Option.none()

    def _build_execution_unit(
        self, chain: ChainBase, context: AgentContext, instructions: str, score: int
    ) -> ExecutionUnit:
        return ExecutionUnit(
            chain,
            context.budget,
            initial_state=ChatMessage.chain(message=instructions) if chain.is_supporting_instructions else None,
            name=f"{chain.name};{score}",
            rank=self.default_execution_unit_rank,
        )
