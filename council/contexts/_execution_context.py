"""

A module providing the ExecutionContext class which holds contextual data for execution logs, including managing current execution paths and associated nodes.

Classes:
    ExecutionContext: A representation of the current context within execution logging.

Attributes:
    _executionLog (ExecutionLog): Private instance of the execution log containing all logged entries.
    _entry (ExecutionLogEntry): Private instance of the current execution log entry.

Methods:
    __init__: Constructor for the ExecutionContext class.
    _new_path: Helper method to construct a new execution path string based on the current path and a new name.
    new_from_name: Creates a new ExecutionContext with a path extended by the given name.
    new_for: Creates a new ExecutionContext for a monitored object, extending the path with the monitored object's name.
    entry: Property to get the current ExecutionLogEntry.
    execution_log: Property to get the ExecutionLog.

Each ExecutionContext instance is designed to encapsulate the state and progress of an execution stream, allowing for fine-grained tracking and logging of operations and their corresponding monitoring data.


"""
from __future__ import annotations
from typing import Optional

from ._monitorable import Monitorable
from ._execution_log import ExecutionLog
from ._execution_log_entry import ExecutionLogEntry
from ._monitored import Monitored


class ExecutionContext:
    """
    Class that retains the execution context for a running process, keeping a log of activities.
    This class is responsible for maintaining a consistent execution context throughout a process,
    enabling tracking and monitoring of the steps carried out during execution. Each instance
    of the class keeps track of its own execution path and the associated execution log.
    
    Attributes:
        _executionLog (ExecutionLog):
             A log object that holds the records of execution steps.
        _entry (ExecutionLogEntry):
             An entry object that represents a specific execution point within the log.
    
    Methods:
        __init__:
             Constructor for ExecutionContext, initializes the log and entry based on provided arguments.
        _new_path:
             Helper method to concatenate execution path names.
        new_from_name:
             Creates a new ExecutionContext using a provided name to extend the execution path.
        new_for:
             Creates a new ExecutionContext in the context of a monitored object.
        entry:
             Property to access the current ExecutionLogEntry instance.
        execution_log:
             Property to access the ExecutionLog associated with this execution context.
    
    Args:
        execution_log (Optional[ExecutionLog], optional):
             The existing execution log to which new entries will be added.
            If None is provided, a new ExecutionLog object will be created.
            Defaults to None.
        path (str, optional):
             The current execution path represented as a string. Defaults to an empty string.
        node (Optional[Monitorable], optional):
             The object that is being monitored, whose inner logic will be logged.
            Defaults to None.
    
    Returns:
        An instance of ExecutionContext with the configured logging and path tracking setup.

    """

    _executionLog: ExecutionLog
    _entry: ExecutionLogEntry

    def __init__(
        self, execution_log: Optional[ExecutionLog] = None, path: str = "", node: Optional[Monitorable] = None
    ):
        """
        Initializes a new instance with an optional execution log, path, and monitorable node.
        The constructor allows for optionally specifying an existing execution log to append to, as well as
        initializing with a given path and a node that is capable of being monitored (Monitorable). If an execution log
        is not provided, a new one will be created. An entry is then created in the execution log with the provided
        path and node.
        
        Args:
            execution_log (Optional[ExecutionLog]):
                 An existing execution log to which a new entry will be added. If not provided, a new execution log will be created.
            path (str):
                 The path associated with the new entry in the execution log. Defaults to an empty string.
            node (Optional[Monitorable]):
                 The node to be monitored, which will be associated with the new log entry. If not specified, no node will be associated.
        
        Attributes:
            _executionLog (ExecutionLog):
                 The execution log where log entries are recorded.
            _entry (ExecutionLogEntry):
                 The current log entry attributed to this instance.
            

        """
        self._executionLog = execution_log or ExecutionLog()
        self._entry = self._executionLog.new_entry(path, node)

    def _new_path(self, name: str) -> str:
        """
        Generates a new path by concatenating the name with the source path of the _entry attribute if it's not empty.
        
        Args:
            name (str):
                 The name to be appended to the source path.
        
        Returns:
            (str):
                 The concatenated path if the source is not empty, otherwise the name itself.

        """
        return name if self._entry.source == "" else f"{self._entry.source}/{name}"

    def new_from_name(self, name: str) -> ExecutionContext:
        """
        Creates a new ExecutionContext with an updated execution path based on a given name.
        This method generates a new execution context by copying the existing execution log and
        appending the provided name to the current execution path to create a new path. It is typically
        used when you need to keep track of a new context within the same execution flow, such as when
        entering a new function or logical block.
        The child context shares the same execution log as the parent, ensuring that all related
        execution entries are kept within the same log.
        
        Args:
            name (str):
                 The name to append to the current execution path to form the new execution path.
        
        Returns:
            (ExecutionContext):
                 A new ExecutionContext object with an updated path reflecting the name provided.

        """
        return ExecutionContext(self._executionLog, self._new_path(name))

    def new_for(self, monitored: Monitored) -> ExecutionContext:
        """
        Creates a new ExecutionContext for a specific `Monitored` instance.
        This method takes a `Monitored` object and generates a new `ExecutionContext` that reflects the
        execution path of the current context appended with the name of the `Monitored` object. It
        also passes through the `ExecutionLog` of the current context to maintain the execution history.
        
        Args:
            monitored (Monitored):
                 An instance of `Monitored` for which the new `ExecutionContext`
                will be created. This object should have a `name` attribute which will be used to
                extend the current execution path, and an `inner` attribute that represents the
                monitorable aspect of the object which will be logged in the `ExecutionLogEntry`.
        
        Returns:
            (ExecutionContext):
                 A new instance of `ExecutionContext` that includes the updated
                execution path and a link to the `Monitorable` element wrapped within the `Monitored`
                object.
            

        """
        return ExecutionContext(self._executionLog, self._new_path(monitored.name), monitored.inner)

    @property
    def entry(self) -> ExecutionLogEntry:
        """
        Gets the ExecutionLogEntry associated with this object.
        
        Returns:
            (ExecutionLogEntry):
                 The ExecutionLogEntry instance representing an execution log entry.

        """
        return self._entry

    @property
    def execution_log(self) -> ExecutionLog:
        """
        Property that gets the current execution log.
        This property returns the private `_executionLog` attribute which should be an instance
        of the `ExecutionLog` class. The execution log typically contains information about
        the execution of a process or task, such as start and end times, status, and any
        messages or errors that occurred during execution.
        
        Returns:
            (ExecutionLog):
                 An object representing the execution log.
            

        """
        return self._executionLog
