from __future__ import annotations

from typing import Optional

from council.chains import Chain
from council.contexts import ChatMessage
from council.runners import Budget


class ExecutionUnit:
    """
    Represents an execution unit to be executed by an Agent

    Parameters:
        chain(Chain): the chain to be executed
        budget(Budget): the budget granted for this execution
        initial_state(Optional[ChatMessage]): an optional message that will be injected in the chain context
        name(Optional[str]): a unique name for the execution. Defaults to :attr:`Chain.name`
    """

    def __init__(
        self, chain: Chain, budget: Budget, initial_state: Optional[ChatMessage] = None, name: Optional[str] = None
    ):
        self._chain = chain
        self._budget = budget
        self._initial_state = initial_state
        self._name = name or chain.name

    @property
    def chain(self) -> Chain:
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
