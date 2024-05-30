from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from council.utils import read_env_int


class BudgetExpiredException(Exception):
    pass


class Consumption:
    """
    A class representing a consumption measurement with value, unit, and kind information.

    Attributes:
        _value (float): The numeric value of the consumption measurement.
        _unit (str): The unit of measurement for the consumption (e.g., tokens, api_calls, etc.).
        _kind (str): The kind or category of the consumption.

    Methods:
        __init__(value: float, unit: str, kind: str):
            Initializes a Consumption instance with the provided value, unit, and kind.

    """

    def __init__(self, value: float, unit: str, kind: str) -> None:
        """
        Initializes a Consumption instance.

        Args:
            value (float): The numeric value of the consumption measurement.
            unit (str): The unit of measurement for the consumption (e.g., liters, watts, etc.).
            kind (str): The kind or category of the consumption (e.g., water, electricity, etc.).

        """
        self._value = value
        self._unit = unit
        self._kind = kind

    @property
    def value(self) -> float:
        return self._value

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def kind(self) -> str:
        return self._kind

    def __str__(self) -> str:
        return f"{self._kind} consumption: {self._value} {self.unit}"

    def add(self, value: float) -> Consumption:
        """
        Returns a new Consumption instance with the value incremented by the specified value.
        """
        return Consumption(self._value + value, self.unit, self._kind)

    def subtract(self, value: float) -> Consumption:
        """
        Returns a new Consumption instance with the value decremented by the specified value.
        """
        return Consumption(self._value - value, self.unit, self._kind)

    def add_value(self, value: float) -> None:
        """
        Increments the value of the consumption by the specified value.
        """
        self._value += value

    def subtract_value(self, value: float) -> None:
        """
        Decrements the value of the consumption by the specified value.
        """
        self._value -= value

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "unit": self.unit, "value": self.value}


class Budget:
    """
    A class representing a budget with duration, limits, and consumption events.

    """

    def __init__(self, duration: float, limits: Optional[List[Consumption]] = None) -> None:
        """
        Initialize the Budget object

        Args:
            duration (float): The duration of the budget in some time unit (e.g., days, months, etc.).
            limits (List[Consumption]): Optional. A list of Consumption objects representing the budget limits.
                                        Each Consumption object contains a value, unit, and kind.

        """
        self._duration = duration
        self._deadline = time.monotonic() + duration
        self._limits = []
        if limits is not None:
            for limit in limits:
                self._limits.append(Consumption(limit.value, limit.unit, limit.kind))

        self._remaining = limits if limits is not None else []

    @property
    def duration(self) -> float:
        """
        the initial duration of the budget
        """
        return self._duration

    @property
    def deadline(self) -> float:
        """
        the deadline of the budget, which when the budget expires
        """
        return self._deadline

    @property
    def remaining_duration(self) -> float:
        """
        the remaining duration in the budget
        """
        return self._deadline - time.monotonic()

    def is_expired(self) -> bool:
        """
        Check if the budget is expired
        Returns:
            True is the budget is expired. Otherwise False
        """
        if self._deadline < time.monotonic():
            return True

        return any(limit.value < 0.0 for limit in self._remaining)

    def can_consume(self, value: float, unit: str, kind: str) -> bool:
        """
        returns `True` if the given consumption is allowed (will not exhaust the budget).
        `False` otherwise
        """
        for limit in self._remaining:
            if limit.unit == unit and limit.kind == kind:
                c = limit.subtract(value)
                return c.value >= 0.0
        return True

    def add_consumption(self, value: float, unit: str, kind: str):
        """
        adds/registers a consumption into the budget
        """
        self._add_consumption(Consumption(value=value, unit=unit, kind=kind))

    def _add_consumption(self, consumption: Consumption) -> None:
        for limit in self._remaining:
            if limit.unit == consumption.unit and limit.kind == consumption.kind:
                limit.subtract_value(consumption.value)

    def add_consumptions(self, consumptions: Iterable[Consumption]) -> None:
        """
        adds/registers many consumptions into the budget
        """
        for consumption in consumptions:
            self._add_consumption(consumption)

    def __repr__(self) -> str:
        return f"Budget({self._duration})"

    @staticmethod
    def default() -> Budget:
        """
        Helper function that create a new Budget with a default value.

        Returns:
            Budget
        """
        duration = read_env_int("COUNCIL_DEFAULT_BUDGET", required=False, default=30)
        return Budget(duration=duration.unwrap())


class InfiniteBudget(Budget):
    """
    Helper class representing a budget with no duration and no limits
    """

    def __init__(self) -> None:
        super().__init__(10000)

    def is_expired(self) -> bool:
        return False
