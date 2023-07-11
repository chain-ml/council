from typing import List, Protocol

from council.core import SkillBase, Budget
from council.core.execution_context import SkillMessage, SkillSuccessMessage, SkillContext
from council.llm import LLMBase, LLMMessage


class ReturnMessages(Protocol):
    def __call__(self, context: SkillContext) -> List[LLMMessage]:
        ...


def get_chat_history(context: SkillContext) -> List[LLMMessage]:
    # Convert chat's history and give it to the inner llm
    return LLMMessage.from_chat_messages(context.chatHistory.messages)


def get_last_messages(context: SkillContext) -> List[LLMMessage]:
    if context.iteration.is_some():
        it_ctxt = context.iteration.unwrap()
        msg = LLMMessage.user_message(it_ctxt.value)
        return [msg]
    last_message = context.current.last_message()
    if last_message.is_none():
        return get_chat_history(context)
    msg = LLMMessage.user_message(last_message.unwrap().message)
    return [msg]


class LLMSkill(SkillBase):
    """Skill to interact with an `LLM`."""

    def __init__(
        self,
        llm: LLMBase,
        name: str = "LLMSkill",
        system_prompt: str = "",
        context_messages: ReturnMessages = get_last_messages,
    ):
        """
        Initialize a new instance of LLMSkill.

        Parameters:
            llm (LLMBase): The instance of the LLM (Language Model) to interact with.
            system_prompt (str): Optional system prompt to provide to the language model.
            context_messages (Callable[[ChainContext], List[LLMMessage]]): Optional callable to retrieve
                context messages.

        Returns:
            None
        """

        super().__init__(name=name)
        self._llm = llm
        self._context_messages = context_messages
        self._system_prompt = LLMMessage.system_message(system_prompt)

    def execute(self, context: SkillContext, _budget: Budget) -> SkillMessage:
        """Execute `LLMSkill`."""

        history_messages = self._context_messages(context)

        # Prepend the system prompt
        messages = [self._system_prompt, *history_messages]
        llm_response = self._llm.post_chat_request(messages=messages)

        return SkillSuccessMessage(skill_name=self.name, message=llm_response, data=None)
