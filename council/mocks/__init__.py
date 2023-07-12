import time
import random
from typing import List, Any, Callable, Optional, Protocol

from council.agents import Agent, AgentResult
from council.core import AgentContext, Budget, ScorerBase, SkillBase
from council.core.execution_context import (
    ScoredAgentMessage,
    AgentMessage,
    SkillContext,
    SkillMessage,
    SkillSuccessMessage,
)
from council.llm import LLMBase, LLMMessage


class LLMMessagesToStr(Protocol):
    def __call__(self, messages: List[LLMMessage]) -> str:
        ...


def llm_message_content_to_str(messages: List[LLMMessage]) -> str:
    return "\n".join([msg.content for msg in messages])


class MockSkill(SkillBase):
    def __init__(self, name: str = "mock", action: Optional[Callable[[SkillContext, Budget], SkillMessage]] = None):
        super().__init__(name)
        self._action = action if action is not None else self.empty_message

    def execute(self, context: SkillContext, budget: Budget) -> SkillMessage:
        return self._action(context, budget)

    def empty_message(self, context: SkillContext, budget: Budget):
        return self.build_success_message("")

    def set_action_custom_message(self, message: str) -> None:
        self._action = lambda context, budget: self.build_success_message(message)


class MockLLM(LLMBase):
    def __init__(self, action: Optional[LLMMessagesToStr] = None):
        self._action = action

    def _post_chat_request(self, messages: List[LLMMessage], **kwargs: Any) -> str:
        if self._action is not None:
            return self._action(messages)
        return f"{self.__class__.__name__}"

    @staticmethod
    def from_responses(responses: List[str]) -> "MockLLM":
        value = "\n".join([r for r in responses])
        return MockLLM(action=(lambda x: value))

    @staticmethod
    def from_response(response: str) -> "MockLLM":
        return MockLLM(action=(lambda x: response))


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

    def execute(self, context: AgentContext, budget: Budget = Budget.default()) -> AgentResult:
        time.sleep(random.uniform(self.sleep, self.sleep + self.sleep_interval))
        return AgentResult([ScoredAgentMessage(AgentMessage(self.message, self.data), score=self.score)])


class MockErrorAgent(Agent):
    # noinspection PyMissingConstructor
    def __init__(self, exception: Exception = Exception()):
        self.exception = exception

    def execute(self, context: AgentContext, budget: Budget = Budget.default()) -> AgentResult:
        raise self.exception
