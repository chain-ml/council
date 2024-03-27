"""

Module _monitored_budget

This module extends the basic Budget class by adding monitoring and logging capabilities.
It leverages ExecutionLogEntry objects to record consumptions and keep track of
the budget usage over time. MonitoredBudget is a subclass of Budget that
initializes itself with the state of an existing Budget object and an associated
ExecutionLogEntry for logging purposes.

Classes:
    MonitoredBudget(Budget): A subclass of Budget that logs each consumption.

The MonitoredBudget class overrides the private method _add_consumption from its superclass
Budget in order to add the consumption to the ExecutionLogEntry before updating the
remaining consumptions in the budget.


"""
from ._budget import Budget, Consumption
from ._execution_log_entry import ExecutionLogEntry


class MonitoredBudget(Budget):
    """
    A class that extends Budget to monitor the budget consumption through logging mechanisms.
    This class is responsible for overseeing budget usage by incorporating an execution log entry. Each time budget consumption
    is recorded, the corresponding details are logged for monitoring and analysis purposes.
    
    Attributes:
        _log_entry (ExecutionLogEntry):
             An instance of ExecutionLogEntry used to log each budget consumption.
    
    Args:
        log_entry (ExecutionLogEntry):
             The log entry object that tracks consumption.
        budget (Budget):
             A budget instance from which this class will inherit and monitor.
    
    Methods:
        _add_consumption(consumption:
             Consumption):
            Overrides the parent method to add the consumption to the budget and log the action.
    
    Args:
        consumption (Consumption):
             The consumption data to be added and logged.
        

    """
    _log_entry: ExecutionLogEntry

    def __init__(self, log_entry: ExecutionLogEntry, budget: Budget):
        """
        Initializes the object with execution log entry and budget details.
        
        Args:
            log_entry (ExecutionLogEntry):
                 The log entry associated with the execution of the job.
            budget (Budget):
                 The budget object containing details about the remaining duration and remaining resources for the execution.
        
        Raises:
            None.
            This is the constructor for the object and is called when a new instance is created. It passes the remaining
            budget duration and resources to the parent class, and stores the `log_entry` for reference.

        """
        super().__init__(budget.remaining_duration, budget._remaining)
        self._log_entry = log_entry

    def _add_consumption(self, consumption: Consumption):
        """
        Adds a consumption record to the log and updates the consumption tracking.
        This method logs a consumption object to the entry log and then calls
        the superclass method to handle the additional consumption tracking logic.
        
        Args:
            consumption (Consumption):
                 The consumption data to log and track.
            

        """
        self._log_entry.log_consumption(consumption)
        super()._add_consumption(consumption)
