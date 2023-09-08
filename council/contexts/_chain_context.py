import logging
from typing import Iterable, List, Optional

import more_itertools

from council.monitors import Monitored
from ._agent_context import AgentContext
from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._cancellation_token import CancellationToken
from ._chat_history import ChatHistory
from ._chat_message import ChatMessage
from ._composite_message_collection import CompositeMessageCollection
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._message_collection import MessageCollection
from ._message_list import MessageList

logger = logging.getLogger(__name__)


class ChainContext(ContextBase, MessageCollection):
    def __init__(
        self,
        store: AgentContextStore,
        execution_context: ExecutionContext,
        name: str,
        budget: Budget,
        messages: Optional[Iterable[ChatMessage]] = None,
    ):
        super().__init__(store, execution_context, budget)
        self._name = name
        self._current_messages = MessageList()
        self._previous_messages = MessageList(messages)

        self._current_iteration_messages = CompositeMessageCollection([self._previous_messages, self._current_messages])
        self._previous_iteration_messages = CompositeMessageCollection(
            list(self._store.chain_iterations(self._name))[:-1]
        )
        self._all_iteration_messages = CompositeMessageCollection(
            [self._previous_iteration_messages, self._current_iteration_messages]
        )
        self._all_messages = CompositeMessageCollection([self.chat_history, self._all_iteration_messages])

    @property
    def cancellation_token(self) -> CancellationToken:
        return self._store.cancellation_token

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def messages(self) -> Iterable[ChatMessage]:
        return self._all_messages.messages

    @property
    def reversed(self) -> Iterable[ChatMessage]:
        return self._all_messages.reversed

    @property
    def chain_histories(self) -> Iterable[MessageCollection]:
        for item in self._store.iterations:
            chain = item.chains.get(self._name)
            if chain is not None:
                yield chain

    @property
    def current(self) -> MessageCollection:
        """
        Returns the :class:`MessageCollection` for the current execution of a :class:`.Chain`

        Returns:
            MessageCollection: a collection of messages
        """
        return self._current_iteration_messages

    @staticmethod
    def from_agent_context(context: AgentContext, monitored: Monitored, name: str, budget: Optional[Budget] = None):
        context._store.current_iteration.ensure_chain_exists(name)
        return ChainContext(
            context._store, context._execution_context.new_for(monitored), name, budget or Budget.default()
        )

    def fork_for(self, monitored: Monitored, budget: Optional[Budget] = None) -> "ChainContext":
        return ChainContext(
            self._store,
            self._execution_context.new_for(monitored),
            self._name,
            budget or self._budget,
            more_itertools.flatten([self._previous_messages.messages, self._current_messages.messages]),
        )

    def should_stop(self) -> bool:
        if self._budget.is_expired():
            logger.debug('message="stopping" reason="budget expired"')
            return True
        if self.cancellation_token.cancelled:
            logger.debug('message="stopping" reason="cancellation token is set"')
            return True

        return False

    def merge(self, contexts: List["ChainContext"]):
        for context in contexts:
            self._current_messages.add_messages(context._current_messages.messages)

    def append(self, message: ChatMessage):
        self._current_messages.add_message(message)
        self._store.current_iteration.append_to_chain(self._name, message, self._execution_context.entry)

    def extend(self, messages: Iterable[ChatMessage]):
        for message in messages:
            self.append(message)

    @staticmethod
    def from_chat_history(history: ChatHistory, budget: Optional[Budget] = None) -> "ChainContext":
        from ..mocks import MockMonitored

        context = AgentContext.from_chat_history(history)
        context.new_iteration()
        return ChainContext.from_agent_context(context, MockMonitored("mock chain"), "mock chain", budget)

    @staticmethod
    def from_user_message(message: str, budget: Optional[Budget] = None) -> "ChainContext":
        return ChainContext.from_chat_history(ChatHistory.from_user_message(message), budget)

    @staticmethod
    def empty() -> "ChainContext":
        return ChainContext.from_chat_history(ChatHistory())
