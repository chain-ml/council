from __future__ import annotations

from council.chains import Chain
from council.runners import Budget


class ExecutionUnit:
    def __init__(self, chain: Chain, budget: Budget):
        self.chain = chain
        self.budget = budget
