"""

Module `python_code_generation_skill`

This module defines the `PythonCodeGenerationSkill` class which inherits from `LLMSkill`. The skill is responsible
for generating Python code based on instructions provided in natural language. It uses a `LLMBase` instance to
understand the context and create code based on a provided template and additional instructions.

Classes:
    PythonCodeGenerationSkill: A skill that aids in generating Python code.


Attributes:
    - SKILL_NAME (str): The name identifying this specific skill.


Functions:
    __init__(self, llm: LLMBase, code_template: str = '', additional_instructions: str = '')
        Initializes a new instance of the PythonCodeGenerationSkill class.

        Args:
            llm (LLMBase): An instance of LLMBase which will be used to understand the prompts and generate
                responses.
            code_template (str): The initial code template to be used for code generation.
            additional_instructions (str): Any additional instructions that might be needed to guide the code
                generation process.


    build_messages(self, context: SkillContext) -> List[LLMMessage]
        Creates a list of LLMMessage instances from the past and present chat history.

        This function processes chat messages into a format suitable for the skill to generate code. It handles
        error messages and converts chat messages into LLMMessage instances which guide the generation process.

        Args:
            context (SkillContext): The current skill context containing the chat history and other relevant
                information for message processing.

        Returns:
            List[LLMMessage]: The processed list of messages to be used by the skill.


"""
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
    A class that represents a skill for generating Python code based on given templates and additional instructions.
    This class extends LLMSkill and specializes in the creation of code snippets or complete Python scripts by
    providing the underlying large language model (LLM) with a formatted system prompt. It stores a unique skill name,
    receives initial settings such as the code template and additional instructions, and handles the communication
    with the LLM by building appropriate messages.
    
    Attributes:
        SKILL_NAME (str):
             A constant string that defines the unique name for this skill.
    
    Methods:
        __init__(self, llm:
             LLMBase, code_template: str='', additional_instructions: str=''): Initialize the
            skill with optional code template and additional instructions, and sets up the system prompt for the LLM.
        build_messages(self, context:
             SkillContext) -> List[LLMMessage]: Constructs a list of LLMMessage
            objects that represent the current context and history of messages, which is then used by the LLM to
            understand the task at hand.
    
    Args:
        llm (LLMBase):
             The large language model that powers this skill.
        code_template (str, optional):
             A string containing a template or a starting snippet of code that can be used
            as a baseline for code generation. Defaults to an empty string.
        additional_instructions (str, optional):
             A string containing any additional instructions or criteria that
            should influence the code generation. Defaults to an empty string.

    """

    SKILL_NAME: str = "PythonCodeGenSkill"

    def __init__(self, llm: LLMBase, code_template: str = "", additional_instructions: str = ""):
        """
        Initializes an object instance with an LLM, code template, and additional instructions.
        
        Args:
            llm (LLMBase):
                 An instance of a Large Language Model.
            code_template (str, optional):
                 A string template containing code that is to be used in
                the system prompt. Defaults to an empty string.
            additional_instructions (str, optional):
                 Additional instructions to supplement the
                main instruction. Defaults to an empty string.
                This initializer constructs the system prompt by formatting the instructions with
                the given code template and additional instructions, and then calls the parent
                initializer with the Large Language Model, skill name, constructed system prompt,
                and the context messages retrieved from the build_messages method.

        """
        system_prompt = instruction.format(code_template=code_template, additional_instructions=additional_instructions)
        super().__init__(llm, self.SKILL_NAME, system_prompt, context_messages=self.build_messages)

    def build_messages(self, context: SkillContext) -> List[LLMMessage]:
        """
        Builds a list of LLMMessage objects representing the conversation history and current messages. It processes each message within the SkillContext.chat_history and updates the list with assistant's messages from the current context. Additionally, it appends error messages from the current context with an instruction to fix the error to the user.
        
        Args:
            context (SkillContext):
                 The context object containing chat history and current messages.
        
        Returns:
            (List[LLMMessage]):
                 A list of LLMMessage objects representing the constructed conversation history enriched with the current context's messages.

        """
        messages = LLMMessage.from_chat_messages(context.chat_history.messages)
        for message in context.current.messages:
            if message.is_from_source(self.name):
                messages.append(LLMMessage.assistant_message(message.message))
            if message.is_error:
                error_message = message.message
                messages.append(LLMMessage.user_message(f"Update the code to fix this error:\n{error_message}"))

        return messages
