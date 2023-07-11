import time
import random
from typing import List, Any, Callable, Optional

from council.core import Agent, AgentContext, Budget, ScorerBase, AgentResult, SkillBase
from council.core.execution_context import ScoredAgentMessage, AgentMessage, SkillContext, SkillMessage
from council.llm import LLMBase, LLMMessage


class MockSkill(SkillBase):
    def __init__(self, name: str = "mock", action: Optional[Callable[[SkillContext, Budget], SkillMessage]] = None):
        super().__init__(name)
        self._action = action if action is not None else self.empty_message

    def execute(self, context: SkillContext, budget: Budget) -> SkillMessage:
        return self._action(context, budget)

    def empty_message(self, context: SkillContext, budget: Budget):
        return self.build_success_message("")


class MockLLM(LLMBase):
    def __init__(self, response: List[str]):
        self.response = "\n".join(response)

    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> str:
        return self.response


class MockErrorSimilarityScorer(ScorerBase):
    def __init__(self, exception: Exception = Exception()):
        self.exception = exception

    def _score(self, message: AgentMessage) -> float:
        raise self.exception


class MockAgent(Agent):
    # noinspection PyMissingConstructor
    def __init__(
        self,
        message: str = "agent message",
        data: Any = None,
        score: float = 1.0,
        sleep: float = 0.2,
        sleep_interval: float = 0.1,
    ):
        self.message = message
        self.data = data
        self.score = score
        self.sleep = sleep
        self.sleep_interval = sleep_interval

    def execute(self, context: AgentContext, budget: Budget) -> AgentResult:
        time.sleep(random.uniform(self.sleep, self.sleep + self.sleep_interval))
        return AgentResult([ScoredAgentMessage(AgentMessage(self.message, self.data), score=self.score)])


class MockErrorAgent(Agent):
    # noinspection PyMissingConstructor
    def __init__(self, exception: Exception = Exception()):
        self.exception = exception

    def execute(self, context: AgentContext, budget: Budget) -> AgentResult:
        raise self.exception
