"""

Module __init__.

This module is responsible for initializing the package and making core classes available for use.
It imports and exposes the `SkillBase`, `LLMSkill`, and `PromptToMessages` classes.

Classes:
    SkillBase: An abstract base class for creating skills for the SkillRunnerBase system.
    LLMSkill: A concrete implementation of `SkillBase`, utilizing a language model (LLM) to execute skills.
    PromptToMessages: A utility class to convert a prompt into a list of LLM messages.


"""

from .skill_base import SkillBase
from .llm_skill import LLMSkill, PromptToMessages
