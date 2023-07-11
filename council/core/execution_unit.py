from __future__ import annotations

from council.core.budget import Budget
from council.core.chain import Chain


class ExecutionUnit:
    def __init__(self, chain: Chain, budget: Budget):
        self.chain = chain
        self.budget = budget
