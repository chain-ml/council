import logging
from typing import List, Optional, Iterable

import more_itertools

from council.contexts import ChainContext, CancellationToken, ChatMessage
from .budget import Budget

logger = logging.getLogger(__name__)


class RunnerContext:
    _chain_context: ChainContext
    _budget: Budget
    _cancellation_token: CancellationToken
    _current_messages: list[ChatMessage]
    _previous_messages: list[ChatMessage]

    def __init__(
        self,
        chain_context: ChainContext,
        budget: Budget,
        messages: Optional[List[ChatMessage]] = None,
    ):
        self._chain_context = chain_context
        self._budget = budget
        self._cancellation_token = chain_context.cancellation_token
        self._current_messages = []
        self._previous_messages = messages or []

    @property
    def chain_context(self) -> ChainContext:
        return self._chain_context

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._cancellation_token

    @property
    def current_messages(self) -> Iterable[ChatMessage]:
        return self._current_messages

    @property
    def previous_messages(self) -> Iterable[ChatMessage]:
        return self._previous_messages

    def append(self, message: ChatMessage):
        if not self.should_stop():
            self._current_messages.append(message)

    @property
    def messages(self) -> Iterable[ChatMessage]:
        return more_itertools.flatten([self._previous_messages, self._current_messages])

    def should_stop(self) -> bool:
        if self._budget.is_expired():
            logger.debug('message="stopping" reason="budget expired"')
            return True
        if self._cancellation_token.cancelled:
            logger.debug('message="stopping" reason="cancellation token is set"')
            return True

        return False

    def make_chain_context(self) -> ChainContext:
        current = self._chain_context.current.copy()
        current.extend(self.messages)
        histories = [*self._chain_context.chain_histories[:-1], current]
        return ChainContext(self._chain_context.chat_history, histories)

    def fork(self) -> "RunnerContext":
        return RunnerContext(
            self._chain_context,
            self._budget.remaining(),
            self._previous_messages + self._current_messages,
        )

    def merge(self, contexts: List["RunnerContext"]):
        for context in contexts:
            self._current_messages.extend(context._current_messages)
