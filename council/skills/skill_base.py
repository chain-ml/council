from __future__ import annotations

import logging
from typing import Any
from abc import abstractmethod

from council.core.budget import Budget
from council.core.execution_context import (
    SkillMessage,
    SkillErrorMessage,
    SkillSuccessMessage,
    SkillContext,
)
from council.core.runners import SkillRunnerBase, RunnerSkillError

logger = logging.getLogger(__name__)


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
    def execute(self, context: SkillContext, budget: Budget) -> SkillMessage:
        """
        Executes the skill on the provided chain context and budget.

        Args:
            context (SkillContext): The context for executing the skill.
            budget (Budget): The budget for skill execution.

        Returns:
            SkillMessage: The result of skill execution.

        Raises:
            None
        """
        pass

    def build_success_message(self, message: str, data: Any = None) -> SkillSuccessMessage:
        """
        Builds a success message for the skill with the provided message and optional data.

        Args:
            message (str): The success message.
            data (Any, optional): Additional data to include in the message. Defaults to None.

        Returns:
            SkillSuccessMessage: The success message.

        Raises:
            None
        """
        return SkillSuccessMessage(self._name, message, data)

    def build_error_message(self, message: str, data: Any = None) -> SkillErrorMessage:
        return SkillErrorMessage(self._name, message, data)

    def from_exception(self, exception: Exception) -> SkillErrorMessage:
        return self.build_error_message(f"skill '{self._name}' raised exception: {exception}")

    def execute_skill(self, context: SkillContext, budget: Budget) -> None:
        try:
            logger.info(f'message="skill execution started" skill="{self.name}"')
            skill_message = self.execute(context, budget)
            if skill_message.is_ok():
                logger.info(
                    f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
                )
            else:
                logger.warning(
                    f'message="skill execution ended" skill="{self.name}" skill_message="{skill_message.message}"'
                )
        except Exception as e:
            logger.exception("unexpected error during execution of skill %s", self.name)
            skill_message = self.from_exception(e)
            raise RunnerSkillError(f"an unexpected error occurred in skill {self.name}") from e
        finally:
            if not self.should_stop(context, budget):
                context.current.messages.append(skill_message)

    def __repr__(self):
        return f"SkillBase({self.name})"

    def __str__(self):
        return f"Skill {self.name}"
