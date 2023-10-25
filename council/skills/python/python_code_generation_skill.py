from __future__ import annotations

from council.contexts import ChatMessage, SkillContext
from council.llm import LLMBase, LLMMessage, MonitoredLLM
from council.skills import SkillBase

from .llm_helper import extract_code_block, LLMMessageParseException

instruction = """
# Instructions
You are a coding assistant generating python code.
- Implement the code using the provided code template respecting
- Your code MUST exactly follow the provided Code Template
- You must never return the code template
{additional_instructions}

# Code Template
```python
{code_template}
```
"""


class PythonCodeGenerationSkill(SkillBase):
    """
    Skill that uses an LLM to generate python code from instructions and a code template.

    The resulting python code is located in a python code block in the resulting message. e.g.::

        Here is the python code I have generated for you

        ```python
        print('hi')
        ```

    """

    SKILL_NAME: str = "PythonCodeGenSkill"

    def __init__(self, llm: LLMBase, code_template: str = "", additional_instructions: str = ""):
        super().__init__(self.SKILL_NAME)
        self._llm = self.register_monitor(MonitoredLLM("llm", llm))
        self._system_prompt = instruction.format(
            code_template=code_template, additional_instructions=additional_instructions
        )

    def execute(self, context: SkillContext) -> ChatMessage:
        messages = [LLMMessage.system_message(self._system_prompt)]

        messages.extend(LLMMessage.from_chat_messages(context.chat_history.messages))

        for message in context.current.messages:
            if message.is_from_source(self.name):
                messages.append(LLMMessage.assistant_message(message.message))
            if message.is_error:
                error_message = (message.data or {}).get(self.SKILL_NAME, None)
                if error_message is None:
                    error_message = message.message
                messages.append(LLMMessage.user_message(f"Update the code to fix this error:\n{error_message}"))

        result = self._llm.post_chat_request(context, messages)
        context.logger.debug(f"llm_response: \n{result.first_choice}")

        try:
            _ = extract_code_block(result.first_choice, "python")
            return self.build_success_message(result.first_choice)
        except LLMMessageParseException as e:
            error = f"{e.__class__.__name__}: {e}"
            context.logger.debug(f"invalid code:\n{error}")
            return self.build_error_message(result.first_choice, {self.SKILL_NAME: error})
