from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._chat_message import ScoredChatMessage
from ._context_base import ContextBase
from ._execution_context import ExecutionContext
from ._message_collection import MessageCollection
from ._monitored import Monitored


class AgentContext(ContextBase):
    """
    the execution context given to an :class:`~council.agents.Agent`
    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget) -> None:
        super().__init__(store, execution_context, budget)

    @staticmethod
    def empty(budget: Optional[Budget] = None) -> AgentContext:
        """
        creates a new instance with no data

        Args:
            budget (Budget): Optional, budget allocated for the agent execution
        """
        return AgentContext.from_chat_history(ChatHistory(), budget)

    @staticmethod
    def from_chat_history(chat_history: ChatHistory, budget: Optional[Budget] = None) -> AgentContext:
        """
        creates a new instance from a :class:`ChatHistory`

        Args:
            chat_history (ChatHistory): The chat history to initialize the new agent context
            budget (Budget): Optional, budget allocated for the agent execution
        """
        store = AgentContextStore(chat_history)
        return AgentContext(store, ExecutionContext(store.execution_log, "agent"), budget or Budget.default())

    @staticmethod
    def from_user_message(message: str, budget: Optional[Budget] = None) -> AgentContext:
        """
        creates a new instance from a user message.
        The :class:`ChatHistory` contains only the given message

        Args:
            message: the user message to start with
            budget (Budget): Optional, budget allocated for the agent execution
        """
        return AgentContext.from_chat_history(ChatHistory.from_user_message(message), budget)

    def new_agent_context_for(self, monitored: Monitored) -> AgentContext:
        """
        creates a new instance for the given object, adjusting the execution context appropriately

        Args:
            monitored: the object to create a new context for
        """
        return AgentContext(self._store, self._execution_context.new_for(monitored), self._budget)

    def new_iteration(self) -> None:
        """
        creates a new execution iteration
        """
        self._store.new_iteration()

    def new_agent_context_for_new_iteration(self) -> AgentContext:
        """
        creates a new instance, adjusting the execution context appropriately
        """
        self.new_iteration()
        name = f"iterations[{len(self._store.iterations) - 1}]"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    def new_agent_context_for_execution_unit(self, name: str) -> AgentContext:
        """
        creates a new instance, adjusting the execution context for the given name

        Args:
            name: name used in the :class:`ExecutionContext`
        """
        name = f"execution({name})"
        return AgentContext(self._store, self._execution_context.new_from_name(name), self._budget)

    @property
    def chains(self) -> Iterable[MessageCollection]:
        """
        provides read-only access to the messages for each chain executed in the current iteration.
        """
        return self._store.current_iteration.chains.values()

    @property
    def evaluation(self) -> Sequence[ScoredChatMessage]:
        """
        provides read-only access to the evaluation of the current iteration
        """
        return self._store.current_iteration.evaluator

    def set_evaluation(self, messages: Iterable[ScoredChatMessage]) -> None:
        """
        sets the evaluation result for the current execution iteration

        Args:
            messages(Iterable[ScoredChatMessage]): the result of the evaluation
        """
        self._store.current_iteration.set_evaluator(messages)
