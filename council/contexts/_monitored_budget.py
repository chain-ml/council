from ._budget import Budget, Consumption
from ._execution_log_entry import ExecutionLogEntry


class MonitoredBudget(Budget):

    def __init__(self, log_entry: ExecutionLogEntry, budget: Budget) -> None:
        super().__init__(budget.remaining_duration, budget._remaining)
        self._log_entry: ExecutionLogEntry = log_entry

    def _add_consumption(self, consumption: Consumption) -> None:
        self._log_entry.log_consumption(consumption)
        super()._add_consumption(consumption)
