from __future__ import annotations

from typing import Any, Iterable

from council.utils import Option

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chain_context import ChainContext
from ._chat_message import ChatMessage
from ._execution_context import ExecutionContext


class IterationContext:
    """
    Provides context information when running inside a loop.
    """

    def __init__(self, index: int, value: Any) -> None:
        self._index = index
        self._value = value

    @property
    def index(self) -> int:
        """
        Returns the index of the current iteration

        Returns:
            int:
        """
        return self._index

    @property
    def value(self) -> Any:
        """
        Returns the value for the current iteration

        Returns:
            Any:
        """
        return self._value

    @staticmethod
    def empty() -> Option[IterationContext]:
        return Option.none()

    @staticmethod
    def new(index: int, value: Any) -> Option[IterationContext]:
        return Option.some(IterationContext(index, value))


class SkillContext(ChainContext):
    """
    Class representing the execution context of a :class:`.SkillBase`.
    """

    def __init__(
        self,
        store: AgentContextStore,
        execution_context: ExecutionContext,
        name: str,
        budget: Budget,
        messages: Iterable[ChatMessage],
        iteration: Option[IterationContext],
    ) -> None:
        super().__init__(store, execution_context, name, budget, messages)
        self._iteration = iteration

    @property
    def iteration(self) -> Option[IterationContext]:
        """
        The iteration context, if any.

        Returns:
            Option[IterationContext]: Some iteration context, if any, else :meth:`.Option.none`
        """
        return self._iteration

    @staticmethod
    def from_chain_context(context: ChainContext, iteration: Option[IterationContext]) -> SkillContext:
        return SkillContext(
            context._store,
            context._execution_context,
            context._name,
            context.budget,
            context.current.messages,
            iteration,
        )
