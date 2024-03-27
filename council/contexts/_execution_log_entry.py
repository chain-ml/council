"""


Module `_execution_log_entry`

This module provides functionality to log execution details such as start time,
duration, consumed resources, messages, and error information during runtime.

Classes:
    ExecutionLogEntry: A class that encapsulates the details of a runtime execution log.

Imports:
    datetime (class): From datetime module, used to record timestamps.
    timezone (class): From datetime module, used for timezone-aware datetimes.
    Any (type): From typing module, represents any type.
    Dict (type): From typing module, for dictionaries with string keys and values of any type.
    List (type): From typing module, for lists containing elements of any type.
    Optional (type): From typing module, denotes an optional type which can be None.
    Sequence (type): From typing module, represents sequence types like lists or tuples.
    Tuple (type): From typing module, for fixed-size immutable sequences.
    Monitorable (class): From the `_monitorable` module, base class for entities that can be monitored.
    Consumption (class): From the `_budget` module, to log resource consumption details.
    ChatMessage (class): From the `_chat_message` module, to log chat messages.

Attributes:
    Not applicable to this module.

Functions:
    Not applicable to this module.



"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._monitorable import Monitorable
from ._budget import Consumption
from ._chat_message import ChatMessage


class ExecutionLogEntry:
    """
    A class representing an execution log entry, which keeps track of various log information for a process.
    This class contains methods for logging different types of messages and consuming
    operations, as well as storing information about a monitorable node if provided.
    It can be used to conveniently manage and export log data in a dictionary format.
    
    Attributes:
        _source (str):
             The source of the log entry, usually indicating where the log originates from.
        _node (Optional[Monitorable]):
             An optional monitorable node associated with the log entry.
        _start (datetime):
             The UTC timestamp when the logging started.
        _duration (float):
             The duration of the process being monitored, in seconds.
        _error (Optional[Exception]):
             An error if one was raised during the monitored process.
        _consumptions (List[Consumption]):
             A list of consumption objects detailing resource usage.
        _messages (List[ChatMessage]):
             A list of chat messages associated with the log entry.
        _logs (List[Tuple[datetime, str, str]]):
             A list of tuples containing log messages with timestamps and level.
    
    Methods:
        __init__:
             Constructor for initializing the log entry with source and node.
        __enter__:
             Context manager entry method, records the start timestamp.
        __exit__:
             Context manager exit method, computes the log duration and records any errors.
        __repr__:
             Represents the log entry as a string.
        to_dict:
             Converts the log entry to a dictionary format suitable for export or serialization.
        log_debug:
             Logs a message with 'DEBUG' level.
        log_info:
             Logs a message with 'INFO' level.
        log_warning:
             Logs a message with 'WARNING' level.
        log_error:
             Logs a message with 'ERROR' level.
        log_consumption:
             Records a single consumption in the log.
        log_consumptions:
             Records multiple consumptions in the log sequentially.
        log_message:
             Records a single chat message in the log.
        _logs_to_dict:
             A helper method to convert the internal log messages to a dictionary format.
    
    Note:
        This class supports the context management protocol, allowing it to be used with the 'with' statement.
        `Monitorable`, `Consumption`, and `ChatMessage` are placeholders for actual classes you might want to monitor
        or log. They need to be defined elsewhere in your codebase.
        

    """

    def __init__(self, source: str, node: Optional[Monitorable]):
        """
        Initializes a new instance with specified source and optional node. It sets the start time to the current time, initializes duration to 0, and creates empty lists for consumptions, messages, and logs.
        
        Args:
            source (str):
                 The source from where the instance is initialized.
            node (Optional[Monitorable]):
                 The monitorable node associated with the instance, if any (default is None).
        
        Attributes:
            _source (str):
                 The source identifier string.
            _node (Optional[Monitorable]):
                 The optional monitorable node linked to this instance. Defaults to None if not provided.
            _start (datetime):
                 The UTC datetime when the instance was initialized.
            _duration (int):
                 The duration of something, initialized to 0 and meant to be specified later.
            _error:
                 Variable for holding error information, if any occurs. Initialized to None.
            _consumptions (List[Consumption]):
                 A list to store consumption details, starts empty.
            _messages (List[ChatMessage]):
                 A list to store chat messages, starts empty.
            _logs (List[Tuple[datetime, str, str]]):
                 A list to store log entries, each entry being a tuple with the time, a string message, and another string, starts empty.

        """
        self._source = source
        self._node = node
        self._start = datetime.now(timezone.utc)
        self._duration = 0
        self._error = None
        self._consumptions: List[Consumption] = []
        self._messages: List[ChatMessage] = []
        self._logs: List[Tuple[datetime, str, str]] = []

    @property
    def source(self) -> str:
        """
        Property that gets the value of the source attribute.
        
        Returns:
            (str):
                 The current value of the _source attribute.

        """
        return self._source

    @property
    def node(self) -> Optional[Monitorable]:
        """
        Gets the current node that is monitorable, if any.
        
        Returns:
            (Optional[Monitorable]):
                 The current monitorable node or None if no node is set.

        """

        return self._node

    def log_consumption(self, consumption: Consumption) -> None:
        """
        Logs a consumption record to the list of consumptions.
        This method takes an instance of Consumption and appends it to the internal
        '_consumptions' list attribute of the class.
        
        Args:
            consumption (Consumption):
                 The Consumption object to be logged.
        
        Returns:
            None
            

        """
        self._consumptions.append(consumption)

    def log_consumptions(self, consumptions: Sequence[Consumption]) -> None:
        """
        Logs each consumption in a sequence of consumptions by delegating to the `log_consumption` method for individual logging actions.
        
        Args:
            consumptions (Sequence[Consumption]):
                 A sequence of Consumption objects to be logged.
        
        Returns:
            (None):
                 This method does not return anything.

        """
        for consumption in consumptions:
            self.log_consumption(consumption)

    def log_message(self, message: ChatMessage) -> None:
        """
        Logs a chat message to the message list of the current instance. The message is appended to the internal _messages list attribute of the instance, effectively storing the message for later retrieval or processing. The method does not return any value. This method is designed for use within a class that has a _messages attribute, which should be a list where instances of ChatMessage are stored. Note the use of the 'self' parameter, which is a reference to the current instance of the class, and the type hint for the message parameter, which is expected to be an instance of ChatMessage or a compatible type. Args: message (ChatMessage): The chat message object to log. Returns: None.

        """
        self._messages.append(message)

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        The __enter__ method is, by convention, part of the context management protocol (the with statement).
        When entering the with statement, this method is invoked. It allows the object to perform any
        setup operations and return itself or another resource.
        
        Returns:
            (object):
                 The resource to be used in the context. The convention is to return the
                object itself, but it can also return a different object that will be used
                inside the with statement's block.

        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Special method facilitating context manager protocol, invoked upon exiting the 'with' block.
        This method is called when exiting the context of a 'with' statement, and it calculates the total duration of time spent within the context in seconds, which it then assigns to the '_duration' attribute. Additionally, it captures any exceptions that occur within the 'with' block by assigning the exception value to the '_error' attribute.
        
        Args:
            exc_type (type, optional):
                 The type of the exception that occurred in the 'with' block, if any. Defaults to None.
            exc_val (Exception, optional):
                 The exception instance that occurred, if an exception has been raised within the 'with' block. Defaults to None.
            exc_tb (traceback, optional):
                 A traceback object representing the point in the code at which the exception occurred, if an exception was raised within the 'with' block. Defaults to None.
        
        Returns:
            (bool):
                 A boolean indicating whether the exception was handled successfully within this method. Typically, returns False unless specifically overridden to return True, which would suppress the exception.

        """
        self._duration = (datetime.now(timezone.utc) - self._start).total_seconds()
        self._error = exc_val

    def __repr__(self):
        """
        Representation function to provide a formal string representation of ExecutionLogEntry.
        This method returns a string that would be a valid Python expression to recreate an
        object with the same data, typically used for debugging.
        
        Returns:
            (str):
                 A formal string representation of the ExecutionLogEntry object including its
                source, start time, duration, and error status.
            

        """
        return (
            "ExecutionLogEntry("
            f"source={self._source}, start={self._start}, duration={self._duration}, error={self._error}"
            ")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Generates a dictionary representation of the object with details about various attributes such as source, start time, duration, consumptions, messages, and logs. It also includes information about any errors that occurred and the current node state, if applicable. This representation can be helpful for serialization or debugging purposes.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing the core data of the object where:
            (- 'source'):
                 a string representing where the object data comes from
            (- 'start'):
                 an ISO formatted string indicating when the process started
            (- 'duration'):
                 a time delta representing how long the process took
            (- 'consumptions'):
                 a list of dictionaries representing consumed items
            (- 'messages'):
                 a list of messages related to the process, serialized to dictionaries
            (- 'logs'):
                 a dictionary representing the logs associated with the process
            (If an error has occurred, the error information will be included under the 'error' key in the format of 'ErrorClass):
                 ErrorMessage'.
                If the object is associated with a node, the node's details will be under the 'node' key, excluding the child node information.

        """
        result = {
            "source": self._source,
            "start": self._start.isoformat(),
            "duration": self._duration,
            "consumptions": [item.to_dict() for item in self._consumptions],
            "messages": [item.to_dict() for item in self._messages],
            "logs": self._logs_to_dict(),
        }

        if self._error is not None:
            result["error"] = f"{self._error.__class__.__name__}: {self._error}"

        if self._node is not None:
            result["node"] = self._node.render_as_dict(include_children=False)

        return result

    def _log_message(self, level: str, message: str, *args: Any) -> None:
        """
        Logs a message with a specific level, also incorporates additional arguments for message formatting.
        This method formats the message string using the provided arguments, if any, and then appends the formatted message along with
        the current timestamp in UTC and the specified log level to the instance's log list.
        
        Args:
            level (str):
                 The level of the log message (e.g., 'INFO', 'DEBUG', 'ERROR').
            message (str):
                 The log message to be stored. This can be a format string that will be interpolated with additional arguments provided.
            *args (Any):
                 Variable length argument list to be used for message string formatting.
        
        Returns:
            None
            

        """
        msg = message % args if len(args) > 0 else message
        self._logs.append((datetime.now(timezone.utc), level, msg))

    def log_debug(self, message: str, *args: Any) -> None:
        """
        Logs a debug message with optional additional arguments.
        This method formats and sends a message to the logging system with a 'DEBUG' level.
        It supports variable arguments to include in the debug message.
        
        Args:
            message (str):
                 The main message to be logged.
            *args (Any):
                 Additional arguments to be formatted within the message.
        
        Returns:
            (None):
                 No value is returned from this method.
            

        """
        self._log_message("DEBUG", message, *args)

    def log_info(self, message: str, *args: Any) -> None:
        """
        Logs an informational message to the defined logging system.
        This method logs a message with an 'INFO' level to the system. It formats the message
        by interpolating any additional arguments provided into the message string.
        
        Args:
            message (str):
                 The message string to log. This can contain format specifiers that
                will be replaced by the values in 'args'.
            args (Any):
                 Additional arguments to interpolate into 'message'. These are passed
                in a way similar to how string formatting works. For example, if the message
                contains '{0}' or '{}', args[0] will replace that placeholder.
        
        Returns:
            (None):
                 This method returns nothing.
            

        """
        self._log_message("INFO", message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        """
        Logs a warning message with the given arguments.
        
        Args:
            message (str):
                 The message template to log, possibly containing format placeholders.
            *args (Any):
                 Optional arguments that are used for message interpolation.
        
        Returns:
            (None):
                 This method does not return anything.
                This method formats the 'message' string with the provided 'args' using Python's standard string formatting mechanism and logs it with a 'WARNING' level. The actual logging mechanism is handled by the '_log_message' method of the class, which this method delegates to with the appropriate log level.

        """
        self._log_message("WARNING", message, *args)

    def log_error(self, message: str, *args: Any) -> None:
        """
        Logs an error message to the system with the given arguments.
        This function logs an error message with an optional number of additional arguments,
        which provides a flexible interface to include various details without needing to
        pre-format the message string.
        
        Args:
            message (str):
                 The main error message to be logged.
            *args (Any):
                 Variable length argument list to be included in the error log.
        
        Returns:
            None.
            

        """
        self._log_message("ERROR", message, *args)

    def _logs_to_dict(self) -> List[Dict[str, Any]]:
        """
        Converts the log entries stored within the instance into a list of dictionaries.
        This internal method iterates over the log entries which are in the form of tuples, consisting of a timestamp,
        a log level, and a log message. It processes each entry and converts it to a dictionary with corresponding
        keys "time", "level", and "message", transforming the timestamp to an ISO formatted string.
        
        Returns:
            (List[Dict[str, Any]]):
                 A list of dictionaries where each dictionary represents a log entry. The keys of the
                dictionary are 'time' for the timestamp in ISO format, 'level' for the log level, and
                'message' for the log message. Each key maps to a value of appropriate type, with 'time' being
                a string, 'level' representing the logged level, and 'message' containing the log text.
            

        """
        return [{"time": item[0].isoformat(), "level": item[1], "message": item[2]} for item in self._logs]
