from ._budget import Budget, Consumption
from ._execution_log_entry import ExecutionLogEntry


class MonitoredBudget(Budget):
    _log_entry: ExecutionLogEntry

    def __init__(self, log_entry: ExecutionLogEntry, budget: Budget):
        super().__init__(budget.remaining_duration, budget._remaining)
        self._log_entry = log_entry

    def add_consumption(self, value: float, unit: str, kind: str):
        consumption = Consumption(value=value, unit=unit, kind=kind)
        self._log_entry.log_consumption(consumption)
        super()._add_consumption(consumption)
