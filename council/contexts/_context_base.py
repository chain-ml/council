"""

Module that provides the base class and utilities for execution context management in an agent system.

This module defines the ContextBase class which acts as the base for creating execution contexts during the running
of agent operations. It encapsulates the agent context store, the execution context, and the budget, providing
a unified interface to log entries, handle budgets, access chat history, and use custom loggers. Additionally,
it provides accessors for the total number of iterations and utility methods to serialize the execution log to
a dictionary or JSON.

Classes:
    ContextBase: The base class for execution contexts that provides essential functionalities required
                 by different parts of an agent system, like logging, budgeting and execution logging.

This module also imports necessary components from other modules to support the execution context features,
including AgentContextStore, Budget, ChatHistory, ContextLogger, ExecutionContext, ExecutionLogEntry,
Monitored, and MonitoredBudget.


"""
from typing import Any, Dict

from ._agent_context_store import AgentContextStore
from ._budget import Budget
from ._chat_history import ChatHistory
from ._context_logger import ContextLogger
from ._execution_context import ExecutionContext
from ._execution_log_entry import ExecutionLogEntry
from ._monitored import Monitored
from ._monitored_budget import MonitoredBudget


class ContextBase:
    """
    A base class for maintaining context within a specific environment for an agent.
    This base class is responsible for handling the storage and retrieval of context-specific
    data during the execution of an agent. It manages the lifecycle of execution contexts,
    budget monitoring, logging, and chat history. The ContextBase class is designed to
    be inherited by other classes that require access to these functionalities.
    
    Attributes:
        _store (AgentContextStore):
             A data store for maintaining agent context details.
        _execution_context (ExecutionContext):
             The current context of execution, tracking
            the execution flow of the agent.
        _budget (MonitoredBudget):
             A wrapped budget object that is monitored to track budget usage.
        _logger (ContextLogger):
             A logger object to record execution logs.
        Properties:
        iteration_count (int):
             The count of iterations stored within the agent context.
        log_entry (ExecutionLogEntry):
             The log entry object of the current execution context.
        budget (Budget):
             The budget available for the current execution context.
        chat_history (ChatHistory):
             The chat history stored within the agent context.
        logger (ContextLogger):
             The logger associated with the current execution context.
    
    Methods:
        __init__:
             Constructs the context base with necessary context components.
        __enter__:
             Enters a runtime context, typically used for resource management.
        __exit__:
             Exits a runtime context, typically used for resource management.
        new_log_entry:
             Creates a new log entry for the given Monitored object.
        execution_log_to_dict:
             Converts the execution log to a dictionary representation.
        execution_log_to_json:
             Serializes the execution log to a JSON-formatted string.
    
    Args:
        store (AgentContextStore):
             The data store for context details.
        execution_context (ExecutionContext):
             The execution context for the current environment.
        budget (Budget):
             The budget available for the agent.
        

    """

    def __init__(self, store: AgentContextStore, execution_context: ExecutionContext, budget: Budget):
        """
        Initializes a new instance of a class with an agent's context store, execution context, and budget monitoring.
        
        Args:
            store (AgentContextStore):
                 The storage mechanism for context-specific data pertinent to an agent.
            execution_context (ExecutionContext):
                 The current state and related data of the running execution context.
            budget (Budget):
                 An object that represents the available resources or limitations, such as time and
                financial constraints, for the current execution context.
        
        Attributes:
            _store (AgentContextStore):
                 A private attribute to hold the passed in store instance.
            _execution_context (ExecutionContext):
                 A private attribute to hold the passed in execution context.
            _budget (MonitoredBudget):
                 A private attribute that wraps the budget with monitoring capabilities,
                logging resource consumption during execution.
            _logger (ContextLogger):
                 A private attribute for logging messages with the execution context's log entry.

        """
        self._store = store
        self._execution_context = execution_context
        self._budget = MonitoredBudget(execution_context.entry, budget)
        self._logger = ContextLogger(execution_context.entry)

    @property
    def iteration_count(self) -> int:
        """
        Property that gets the number of iterations stored.
        This property counts the number of iterations that have been stored by
        accessing the length of the iterations container within the private `_store` attribute.
        
        Returns:
            (int):
                 The total number of iterations currently stored.
            

        """
        return len(self._store.iterations)

    @property
    def log_entry(self) -> ExecutionLogEntry:
        """
        Gets the current execution log entry. This is a property of the class that, when accessed, returns the
        log entry associated with the execution context of this instance.
        
        Returns:
            (ExecutionLogEntry):
                 The log entry corresponding to the current execution context.

        """
        return self._execution_context.entry

    @property
    def budget(self) -> Budget:
        """
        
        Returns the budget associated with the current instance.

        """
        return self._budget

    @property
    def chat_history(self) -> ChatHistory:
        """
        
        Returns the chat history associated with this instance.
            Chat history includes the records of previous conversations that can be used
            for various purposes like data analysis, machine learning training, or simply
            to retrieve past conversations.
            (Property):
                This method is a property decorator, which means the chat_history can be accessed
                like an attribute without the need to call it as a function.
        
        Returns:
            An instance of ChatHistory that contains the chat records.
            

        """
        return self._store.chat_history

    @property
    def logger(self) -> ContextLogger:
        """
        A property that returns the ContextLogger instance associated with this class.
        This property allows access to the ContextLogger that is intended to be used for logging purposes within the class. The logger should be a private member (_logger) of the class, ensuring encapsulation and that the logger is accessed in a controlled manner.
        
        Returns:
            (ContextLogger):
                 The ContextLogger instance that is used for logging within the class.
            

        """
        return self._logger

    def __enter__(self):
        """
        Enters the runtime context related to this object which is designed to be used with the `with` statement. The `with` statement will bind the variable after the `as` keyword to the returned object. When entering the context, it invokes the `__enter__` method of the log entry associated with this object as well. This method is part of the context manager protocol.
        
        Returns:
            The instance itself (self), configured and ready to be used in a `with` block.
        
        Raises:
            Whatever exceptions may be raised by the `__enter__` method of the log entry property, which are context-specific to the log entry implementation.

        """
        self.log_entry.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handle the exit from the context manager for this object.
        This method is called when exiting the `with` block. It ensures that any cleanup for the `log_entry` object is handled. If exceptions are raised within the `with` block, `__exit__` receives their types, values, and traceback information, which can be handled or re-raised within this method.
        
        Args:
            exc_type (type):
                 The exception type if raised in the context block, otherwise `None`.
            exc_val (Exception):
                 The exception instance if raised in the context block, otherwise `None`.
            exc_tb (traceback):
                 A traceback object if an exception is raised in the context block, otherwise `None`.
        
        Returns:
            (bool):
                 A boolean indicating whether the exception was properly handled. If `True`, no exception is propagated beyond the context manager. If `False` or `None`, any exception raised in the context block will be re-raised after this method completes.
            

        """
        self.log_entry.__exit__(exc_type, exc_val, exc_tb)

    def new_log_entry(self, monitored: Monitored) -> ExecutionLogEntry:
        """
        Creates a new log entry for a monitored item within the execution context.
        This method generates a new `ExecutionLogEntry` by creating a new monitored execution context
        specific to the `monitored` parameter. It connects the monitored item with an execution context
        and retrieves the `entry` attribute to serve as the log entry.
        
        Args:
            self:
                 The instance through which the method is being invoked.
            monitored (Monitored):
                 The instance representing the item being monitored.
        
        Returns:
            (ExecutionLogEntry):
                 A log entry associated with the given monitored item within the current execution context.
            

        """
        return self._execution_context.new_for(monitored).entry

    def execution_log_to_dict(self) -> Dict[str, Any]:
        """
        Converts the execution log of the current execution context to a dictionary.
        This method retrieves the execution log from the current execution context object, converts it to a dictionary,
        and then returns that dictionary. The dictionary format makes it suitable for serialization or
        processing in contexts where a native Python dictionary is required.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representation of the execution log.
            

        """
        return self._execution_context.execution_log.to_dict()

    def execution_log_to_json(self) -> str:
        """
        Converts the execution log to a JSON string.
        This method serializes the execution log of the current execution context into a
        JSON formatted string. It relies on the `to_json` method of the execution log object.
        
        Returns:
            (str):
                 A JSON string representing the execution log.

        """
        return self._execution_context.execution_log.to_json()
