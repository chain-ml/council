"""

Module: _context_logger

This module provides the `ContextLogger` class for structured logging. It enhances conventional
logging by associating log messages with execution contexts, represented as `ExecutionLogEntry`
instances. The `ContextLogger` allows capturing log messages at different severities and stores
them within the execution log entry for post-processing or analysis. It forwards messages to the
standard `logging` module while maintaining a copy in the associated execution log entry.

Classes:
    ContextLogger: A logging wrapper that ties log entries to an execution context.

Class `ContextLogger`:
    This class serves as a wrapper for the standard Python logging, extending it to handle
    execution-specific log entries. It logs messages both to the Python logging module and
    to an `ExecutionLogEntry` object for centralized collection of logs tied to a particular
    execution flow.

    The `ContextLogger` class supports the typical logging methods like debug, info, warning,
    error, and exception. In addition to logging to the Python's standard logging library, it
    appends the log messages to the curated log list maintained within the provided
    `ExecutionLogEntry` object.

    Methods:
        __init__(self, log_entry: ExecutionLogEntry)
            Initializes the `ContextLogger` with the given `ExecutionLogEntry` object.
        debug(self, message: str, *args: Any) -> None
            Logs a debug message to both the standard logger and the execution log.
        error(self, message: str, *args: Any) -> None
            Logs an error message to both the standard logger and the execution log.
        exception(self, message: str, *args: Any)
            Logs an error message, including stack trace info, to both the standard logger
            and the execution log.
        info(self, message: str, *args: Any) -> None
            Logs an info message to both the standard logger and the execution log.
        warning(self, message: str, *args: Any) -> None
            Logs a warning message to both the standard logger and the execution log.

    Static Methods:
        _logger_log(level: int, message: str, *args: Any, exc_info: bool=False) -> bool
            Forwards a message at the given log level to the Python logger, and returns a
            boolean indicating whether the log level is enabled.



"""
import logging
from typing import Any

from ._execution_log_entry import ExecutionLogEntry


class ContextLogger:
    """
    A utility class `ContextLogger` that acts as a contextual logging wrapper, integrating both standard
    logging and a specific log entry object behavior, such as appending log messages to a database or monitoring system.
    This class provides methods to log messages at different severity levels: debug, info, warning, error,
    and exception. Each of these methods first checks if the corresponding logging level is enabled,
    and then writes out the message using the standard Python logging library. In addition, it records
    the log message to a custom `ExecutionLogEntry` object if the logging level is indeed enabled, which
    is intended to be integrated into a larger logging or monitoring framework.
    
    Attributes:
        _log_entry (ExecutionLogEntry):
             An instance of `ExecutionLogEntry` which this `ContextLogger`
            uses to record log messages in a structured manner.
    
    Methods:
        debug(message:
             str, *args: Any) -> None:
            Logs a debug message.
        error(message:
             str, *args: Any) -> None:
            Logs an error message.
        exception(message:
             str, *args: Any):
            Logs an exception message, with the stack trace.
        info(message:
             str, *args: Any) -> None:
            Logs an informational message.
        warning(message:
             str, *args: Any) -> None:
            Logs a warning message.
        _logger_log(level:
             int, message: str, *args: Any, exc_info: bool=False) -> bool:
            A static method that performs the actual logging using Python's logging module.
            It also determines if the log level is enabled for the logger.

    """
    def __init__(self, log_entry: ExecutionLogEntry):
        """
        Initializes an instance of the containing class with a log entry object. This constructor sets the provided log entry to a protected member variable of the instance for later use. To use this initializer, an object of type ExecutionLogEntry should be provided, which represents a singular entry in an execution log, capturing details relevant to a specific event or state in the application's execution. For instance, the log entry might contain timestamps, event descriptions, or other contextual information that needs to be associated with the instance for logging purposes. Args: log_entry (ExecutionLogEntry): An object representing a single entry in an execution log.

        """
        self._log_entry = log_entry

    def debug(self, message: str, *args: Any) -> None:
        """
        Logs a debug message through the system's logging facility.
        This method logs a message with debug level information if the logger is enabled for the DEBUG level. The message might be formatted with additional variable-length arguments.
        
        Args:
            message (str):
                 The message to log.
            *args (Any):
                 Variable-length argument list used for message formatting.
        
        Returns:
            None

        """
        if self._logger_log(logging.DEBUG, message, *args):
            self._log_entry.log_debug(message, *args)

    def error(self, message: str, *args: Any) -> None:
        """
        Logs an error message and additional arguments to the logger if logging is successful,
        and also records the error message to the log entry.
        The error message is formatted with any additional arguments provided, and both
        the formatted message and raw arguments are passed to the logger's log function.
        If the logger logs the error successfully, the same message is then recorded
        in the log entry with the associated arguments.
        
        Args:
            message (str):
                 The error message to be logged.
            *args (Any):
                 Additional arguments to be formatted within the message.
        
        Returns:
            None
            

        """
        if self._logger_log(logging.ERROR, message, *args):
            self._log_entry.log_error(message, *args)

    def exception(self, message: str, *args: Any):
        """
        Logs an exception with a given message and additional arguments.
        This method handles the logging of an exception by first determining whether the logger can log at
        the ERROR level. If it can, the method logs the error with the provided message and arguments.
        It then proceeds to create an error log entry with the same information. The exception's traceback
        is included in the log to provide more context on the error's origin.
        
        Args:
            message (str):
                 The error message to be logged.
            *args (Any):
                 Additional arguments that should be formatted into the message.
        
        Raises:
            Any exception raised during the logging process is implicitly allowed to propagate,
            as there is no explicit handling of such exceptions within this method.

        """
        if self._logger_log(logging.ERROR, message, *args, exc_info=True):
            self._log_entry.log_error(message, *args)

    def info(self, message: str, *args: Any) -> None:
        """
        Logs an info level message both to the logger and log entry if applicable.
        This method checks if info level logging is enabled through `self._logger_log`.
        If it is, it logs the message using `self._log_entry.log_info`. This function is
        useful for recording informative messages that can help understand the
        flow of execution or state changes in the application.
        
        Args:
            message (str):
                 The message to be logged.
            *args (Any):
                 Variable length argument list that may be used to format the message string.
        
        Returns:
            (None):
                 There is no return value for this method.

        """
        if self._logger_log(logging.INFO, message, *args):
            self._log_entry.log_info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        """
        
        Raises a logging warning with the specified message and arguments.
            This method will log a warning message using the configured logger. If the logging operation returns True,
            meaning the message has been successfully logged, it will also record the warning by calling
            `_log_entry.log_warning()` with the message and additional arguments.
        
        Args:
            message (str):
                 The warning message to be logged.
            *args (Any):
                 Additional arguments to be included in the log entry.

        """
        if self._logger_log(logging.WARNING, message, *args):
            self._log_entry.log_warning(message, *args)

    @staticmethod
    def _logger_log(level: int, message: str, *args: Any, exc_info: bool = False) -> bool:
        """
        Logs a message with the specified level on the root logger.
        This method constructs a logger dynamically based on the context from which it
        is called. The logging level, message, additional arguments, and exception
        info are used to log the message appropriately. It also returns a value
        indicating whether the logger is enabled for the given logging level.
        
        Args:
            level (int):
                 An integer that represents the level of the log message
                (e.g., logging.INFO, logging.DEBUG).
            message (str):
                 The message string to be logged. It supports string
                formatting with additional args.
            *args (Any):
                 Variable length argument list to be interpolated into the
                message using the standard string formatting syntax.
            exc_info (bool, optional):
                 A boolean that indicates whether exception
                information should be added to the logging message. Defaults to False.
        
        Returns:
            (bool):
                 A boolean value that indicates if the logger is enabled for the
                specified logging level.

        """
        import inspect

        stack = inspect.stack()
        logger_name = stack[2].frame.f_globals["__name__"]
        logger = logging.getLogger(logger_name)
        logger.log(level, message, *args, stacklevel=3, exc_info=exc_info)
        return logger.isEnabledFor(level)
