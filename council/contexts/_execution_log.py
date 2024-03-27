"""

Module `_execution_log`.

This module contains the `ExecutionLog` class which is used for recording execution log entries
associated with `Monitorable` nodes. The log entries capture various aspects of execution including
source, duration, any errors that occurred, and additional information like consumption details
and messages. The `ExecutionLog` provides methods to create new entries and serialize the entire
log to a JSON formatted string. The logs generated can be used for debugging, performance
monitoring, or auditing execution flows.

Classes:
    ExecutionLog: A class that encapsulates the creation and storage of execution log entries.

Typical usage example:
    execution_log = ExecutionLog()
    with execution_log.new_entry("operation_name", monitorable_node) as entry:
        # Perform operation
        ...
    log_json = execution_log.to_json()


"""
import json
from typing import Any, Dict, Optional

from ._monitorable import Monitorable
from ._execution_log_entry import ExecutionLogEntry


class ExecutionLog:
    """
    A class that represents a log of execution entries, capable of creating new log entries and converting the log entries to JSON format.
    
    Attributes:
        _entries (List[ExecutionLogEntry]):
             A private list that stores instances of ExecutionLogEntry.
    
    Methods:
        __init__(self):
            Initializes a new instance of ExecutionLog with an empty list of entries.
        new_entry(self, name:
             str, node: Optional[Monitorable]) -> ExecutionLogEntry:
            Creates a new log entry with the provided name and monitorable node, adds it to the log, and returns the newly created log entry.
    
    Args:
        name (str):
             The name of the log entry.
        node (Optional[Monitorable]):
             The monitorable node related to the log entry, which can be None.
    
    Returns:
        (ExecutionLogEntry):
             The newly created log entry instance.
        to_json(self) -> str:
            Converts the execution log entries to a JSON string with indentation for readability.
    
    Returns:
        (str):
             The JSON string representation of the execution log.
        to_dict(self) -> Dict[str, Any]:
            Converts the execution log to a dictionary that can be easily serialized to JSON.
    
    Returns:
        (Dict[str, Any]):
             The dictionary representing the execution log ready for serialization.

    """

    def __init__(self):
        """
        Initializes a new instance of the class, setting up an empty list for entries.
        
        Attributes:
            _entries (List[Any]):
                 A private list that will store the entries.

        """
        self._entries = []

    def new_entry(self, name: str, node: Optional[Monitorable]) -> ExecutionLogEntry:
        """
        Creates and appends a new ExecutionLogEntry to the log entries list.
        This function initializes a new instance of ExecutionLogEntry with the provided 'name' and 'node',
        appends it to the internal list of log entries, and returns the newly created log entry.
        
        Args:
            name (str):
                 The source name from which the execution log entry is being created.
            node (Optional[Monitorable]):
                 The monitorable node associated with the log entry. Can be None.
        
        Returns:
            (ExecutionLogEntry):
                 The newly created and appended execution log entry.

        """
        result = ExecutionLogEntry(name, node)
        self._entries.append(result)
        return result

    def to_json(self) -> str:
        """
        Serializes the object to a JSON-formatted string with indentation.
        This method converts the object's dictionary representation into a JSON-formatted string using a standard
        indentation of 2 spaces for better readability. The conversion is done through the `json.dumps` method from
        Python's standard json library.
        
        Returns:
            (str):
                 A JSON-formatted string representing the object.
            

        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """
        
        Returns a dictionary containing the entries in the collection with their associated data.
            This method goes through each entry in the private `_entries` attribute, calling the `to_dict` method
            on each to convert them to dictionaries, and then constructs a single dictionary to represent
            the entire collection. The dictionary includes one key 'entries', which maps to a list of the
            dictionary representation of each entry.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary with a key 'entries' that contains a list of dictionaries,
                each representing one entry in the collection.

        """
        result = {"entries": [item.to_dict() for item in self._entries]}

        return result
