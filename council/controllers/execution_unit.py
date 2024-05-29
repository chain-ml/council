from __future__ import annotations

from typing import Optional

from council.chains import ChainBase
from council.contexts import Budget, ChatMessage


class ExecutionUnit:
    """
    Represents an execution unit to be executed by an Agent

    Parameters:
        chain(ChainBase): the chain to be executed
        budget(Budget): the budget granted for this execution
        initial_state(Optional[ChatMessage]): an optional message that will be injected in the chain context
        name(Optional[str]): a unique name for the execution. Defaults to :attr:`Chain.name`
        rank(Optional[int]): execution rank for execution, executed by ascending order.
            Same rank are executed in parallel. If not set, default to sequential execution.
    """

    def __init__(
        self,
        chain: ChainBase,
        budget: Budget,
        initial_state: Optional[ChatMessage] = None,
        name: Optional[str] = None,
        rank: Optional[int] = None,
    ) -> None:
        self._chain = chain
        self._budget = budget
        self._initial_state = initial_state
        self._name = name or chain.name
        self._rank = rank or -1

    @property
    def chain(self) -> ChainBase:
        """
        The chain to be executed

        Returns:
            Chain:
        """
        return self._chain

    @property
    def budget(self) -> Budget:
        """
        The budget for the execution

        Returns:
            Budget:
        """
        return self._budget

    @property
    def initial_state(self) -> Optional[ChatMessage]:
        """
        An optional message to put in the chain context

        Returns:
            Optional[ChatMessage]
        """
        return self._initial_state

    @property
    def name(self) -> str:
        """
        Name of the execution unit

        Returns:
            str:
        """
        return self._name

    @property
    def rank(self) -> int:
        """
        Execution Rank

        Returns:
            int:
        """
        return self._rank
