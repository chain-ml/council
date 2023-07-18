from __future__ import annotations

from typing import Optional

from council.chains import Chain
from council.contexts import ChatMessage
from council.runners import Budget


class ExecutionUnit:
    def __init__(
        self, chain: Chain, budget: Budget, initial_state: Optional[ChatMessage] = None, name: Optional[str] = None
    ):
        self._chain = chain
        self._budget = budget
        self._initial_state = initial_state
        self._name = name or chain.name

    @property
    def chain(self) -> Chain:
        return self._chain

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def initial_state(self) -> Optional[ChatMessage]:
        return self._initial_state

    @property
    def name(self) -> str:
        return self._name
