from __future__ import annotations

from typing import List

from council.contexts import SkillContext
from council.llm import LLMBase, LLMMessage
from council.skills import LLMSkill


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


class PythonCodeGenerationSkill(LLMSkill):
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
        system_prompt = instruction.format(code_template=code_template, additional_instructions=additional_instructions)
        super().__init__(llm, self.SKILL_NAME, system_prompt, context_messages=self.build_messages)

    def build_messages(self, context: SkillContext) -> List[LLMMessage]:
        messages = LLMMessage.from_chat_messages(context.chat_history.messages)
        for message in context.current.messages:
            if message.is_from_source(self.name):
                messages.append(LLMMessage.assistant_message(message.message))
            if message.is_error:
                error_message = message.message
                messages.append(LLMMessage.user_message(f"Update the code to fix this error:\n{error_message}"))

        return messages
