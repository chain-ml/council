from ._budget import Budget, Consumption
from ._execution_log_entry import ExecutionLogEntry


class MonitoredBudget(Budget):
    _log_entry: ExecutionLogEntry

    def __init__(self, log_entry: ExecutionLogEntry, budget: Budget):
        super().__init__(budget.remaining_duration, budget._limits, budget._consumptions)
        self._log_entry = log_entry

    def add_consumption(self, consumption: Consumption, source: str):
        self._log_entry.log_consumption(consumption)
        super().add_consumption(consumption, source)
