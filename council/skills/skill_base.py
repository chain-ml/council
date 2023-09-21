from __future__ import annotations

from typing import Any
from abc import abstractmethod

from council.contexts import SkillContext, ChatMessage
from council.runners import SkillRunnerBase


class SkillBase(SkillRunnerBase):
    """
    Abstract base class for a skill.
    """

    _name: str

    def __init__(self, name: str):
        """
        Initializes the Skill object with the provided name.

        Args:
            name (str): The name of the skill.
        Raises:
            None
        """
        super().__init__(name)
        self._name = name

    @property
    def name(self):
        """
        Property getter for the skill name.

        Returns:
            str: The name of the skill.

        Raises:
            None
        """
        return self._name

    @abstractmethod
    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes the skill on the provided chain context and budget.

        Args:
            context (SkillContext): The context for executing the skill.

        Returns:
            ChatMessage: The result of skill execution.

        Raises:
            None
        """
        pass

    def build_success_message(self, message: str, data: Any = None) -> ChatMessage:
        """
        Builds a success message for the skill with the provided message and optional data.

        Args:
            message (str): The success message.
            data (Any, optional): Additional data to include in the message. Defaults to None.

        Returns:
            ChatMessage: The success message.

        Raises:
            None
        """
        return ChatMessage.skill(message, data, source=self._name, is_error=False)

    def build_error_message(self, message: str, data: Any = None) -> ChatMessage:
        return ChatMessage.skill(message, data, source=self._name, is_error=True)

    def execute_skill(self, context: SkillContext) -> ChatMessage:
        context.logger.info(f'message="skill execution started" skill="{self.name}"')
        skill_message = self.execute(context)
        if skill_message.is_ok:
            context.logger.info(
                f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
            )
        else:
            context.logger.warning(
                f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
            )
        return skill_message

    def __repr__(self):
        return f"SkillBase({self.name})"

    def __str__(self):
        return f"Skill {self.name}"
