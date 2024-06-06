import random
import time
from typing import Any, Optional

from council import Agent, AgentContext, AgentResult, Budget, ChatMessage
from council.contexts import ScoredChatMessage


class MockAgent(Agent):
    # noinspection PyMissingConstructor
    def __init__(
        self,
        message: str = "agent message",
        data: Any = None,
        score: float = 1.0,
        sleep: float = 0.2,
        sleep_interval: float = 0.1,
    ) -> None:
        self.message = message
        self.data = data
        self.score = score
        self.sleep = sleep
        self.sleep_interval = sleep_interval

    def execute(self, context: AgentContext, budget: Optional[Budget] = None) -> AgentResult:
        time.sleep(random.uniform(self.sleep, self.sleep + self.sleep_interval))
        return AgentResult([ScoredChatMessage(ChatMessage.agent(self.message, self.data), score=self.score)])


class MockErrorAgent(Agent):
    # noinspection PyMissingConstructor
    def __init__(self, exception: Exception = Exception()) -> None:
        self.exception = exception

    def execute(self, context: AgentContext, budget: Optional[Budget] = None) -> AgentResult:
        raise self.exception
