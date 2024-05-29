import logging
from typing import Any

from ._execution_log_entry import ExecutionLogEntry


class ContextLogger:
    def __init__(self, log_entry: ExecutionLogEntry) -> None:
        self._log_entry = log_entry

    def debug(self, message: str, *args: Any) -> None:
        """
        Logs a debug message using `logging` and keeps track of it into the context

        Args:
            message (str): a message. Could be formatted a `msg % args`
            *args: optional format parameters
        """
        if self._logger_log(logging.DEBUG, message, *args):
            self._log_entry.log_debug(message, *args)

    def error(self, message: str, *args: Any) -> None:
        """
        Logs an error message using `logging` and keeps track of it into the context

        Args:
            message (str): a message. Could be formatted a `msg % args`
            *args: optional format parameters
        """
        if self._logger_log(logging.ERROR, message, *args):
            self._log_entry.log_error(message, *args)

    def exception(self, message: str, *args: Any):
        if self._logger_log(logging.ERROR, message, *args, exc_info=True):
            self._log_entry.log_error(message, *args)

    def info(self, message: str, *args: Any) -> None:
        """
        Logs an info message using `logging` and keeps track of it into the context

        Args:
            message (str): a message. Could be formatted a `msg % args`
            *args: optional format parameters
        """
        if self._logger_log(logging.INFO, message, *args):
            self._log_entry.log_info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        """
        Logs a warning message using `logging` and keeps track of it into the context

        Args:
            message (str): a message. Could be formatted a `msg % args`
            *args: optional format parameters

        """
        if self._logger_log(logging.WARNING, message, *args):
            self._log_entry.log_warning(message, *args)

    @staticmethod
    def _logger_log(level: int, message: str, *args: Any, exc_info: bool = False) -> bool:
        import inspect

        stack = inspect.stack()
        logger_name = stack[2].frame.f_globals["__name__"]
        logger = logging.getLogger(logger_name)
        logger.log(level, message, *args, stacklevel=3, exc_info=exc_info)
        return logger.isEnabledFor(level)
