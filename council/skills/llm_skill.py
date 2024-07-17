from typing import List, Protocol

from council.contexts import ChatMessage, SkillContext
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.prompt import PromptBuilder
from council.skills import SkillBase


class ReturnMessages(Protocol):
    def __call__(self, context: SkillContext) -> List[LLMMessage]: ...


def get_chat_history(context: SkillContext) -> List[LLMMessage]:
    # Convert chat's history and give it to the inner llm
    return LLMMessage.from_chat_messages(context.chat_history.messages)


def get_last_messages(context: SkillContext) -> List[LLMMessage]:
    if context.iteration.is_some():
        it_ctxt = context.iteration.unwrap()
        msg = LLMMessage.user_message(it_ctxt.value)
        return [msg]
    last_message = context.current.try_last_message
    if last_message.is_none():
        return get_chat_history(context)
    msg = LLMMessage.user_message(last_message.unwrap().message)
    return [msg]


class PromptToMessages:
    def __init__(self, prompt_builder: PromptBuilder) -> None:
        self._builder = prompt_builder

    def to_system_message(self, context: SkillContext) -> List[LLMMessage]:
        msg = self._builder.apply(context)
        context.logger.debug(f'prompt="{msg}')
        return [LLMMessage.system_message(msg)]

    def to_user_message(self, context: SkillContext) -> List[LLMMessage]:
        msg = self._builder.apply(context)
        context.logger.debug(f'prompt="{msg}')
        return [LLMMessage.user_message(msg)]


class LLMSkill(SkillBase):
    """Skill to interact with an `LLM`."""

    def __init__(
        self,
        llm: LLMBase,
        name: str = "LLMSkill",
        system_prompt: str = "",
        context_messages: ReturnMessages = get_last_messages,
    ) -> None:
        """
        Initialize a new instance of LLMSkill.

        Parameters:
            llm (LLMBase): The instance of the LLM (Language Model) to interact with.
            system_prompt (str): Optional system prompt to provide to the language model.
            context_messages (Callable[[SkillContext], List[LLMMessage]]): Optional callable to retrieve
                context messages.

        Returns:
            None
        """

        super().__init__(name=name)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._context_messages = context_messages
        self._builder = PromptBuilder(system_prompt)

    @property
    def llm(self) -> LLMBase:
        """
        the LLM used by the skill
        """
        return self._llm.inner

    def execute(self, context: SkillContext) -> ChatMessage:
        """Execute `LLMSkill`."""

        history_messages = self._context_messages(context)
        system_prompt = LLMMessage.system_message(self._builder.apply(context))
        messages = [system_prompt, *history_messages]
        llm_response = self._llm.post_chat_request(context, messages=messages)
        if len(llm_response.choices) < 1:
            return self.build_error_message(message="no response")

        context.budget.add_consumption(1, "call", "LLMSkill")

        return self.build_success_message(message=llm_response.first_choice, data=llm_response)
