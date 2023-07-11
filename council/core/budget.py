from __future__ import annotations

import time


class BudgetExpiredException(Exception):
    pass


class Budget:
    """
    Represents the budget available for the execution

    Attributes:
        duration (float): The number of seconds
    """

    def __init__(self, duration: float):
        """
        Initialize the Budget object

        Args:
            duration (float): The number of seconds
        """
        self.duration = duration
        self.deadline = time.monotonic() + duration

    def remaining(self) -> Budget:
        """
        Create a new instance with the remaining budget

        Returns:
            a new instance with the remaining budget
        """
        return Budget(self.deadline - time.monotonic())

    def is_expired(self) -> bool:
        """
        Check if the budget is expired
        Returns:
            True is the budget is expired. Otherwise False
        """
        return self.deadline < time.monotonic()

    def __repr__(self):
        return f"Budget({self.duration})"


class InfiniteBudget(Budget):
    def __init__(self):
        super().__init__(10000)

    def remaining(self) -> Budget:
        return Budget(10000)

    def is_expired(self) -> bool:
        return False
