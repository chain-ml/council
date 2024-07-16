from __future__ import annotations

import time
from typing import Callable, Optional

from council import ChatMessage, SkillContext
from council.skills import SkillBase


class MockSkill(SkillBase):
    def __init__(self, name: str = "mock", action: Optional[Callable[[SkillContext], ChatMessage]] = None) -> None:
        super().__init__(name)
        self._action = action if action is not None else self.empty_message

    def execute(self, context: SkillContext) -> ChatMessage:
        return self._action(context)

    def empty_message(self, context: SkillContext) -> ChatMessage:
        return self.build_success_message("")

    def set_action_custom_message(self, message: str) -> None:
        self._action = lambda context: self.build_success_message(message)

    @staticmethod
    def build_wait_skill(duration: int = 1, message: str = "done") -> MockSkill:
        def wait_a_message(context: SkillContext) -> ChatMessage:
            time.sleep(duration)
            return ChatMessage.skill(message)

        if duration > 0:
            return MockSkill(action=wait_a_message)
        return MockSkill(action=lambda context: ChatMessage.skill(message))
