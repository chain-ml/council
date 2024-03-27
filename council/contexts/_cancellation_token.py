"""


Module _cancellation_token

This module contains the CancellationToken class which provides mechanisms to handle cancellation signals across threads.

Classes:
    CancellationToken: A thread-safe cancellation token that can be checked and set by different threads to manage cooperative cancellation of operations.



"""
from threading import Lock


class CancellationToken:
    """
    A class to represent a token that can be used to manage cancellation signals across threads or tasks.
    This class provides a simple way to introduce cooperative cancellation in applications by setting
    and checking the cancellation state. It can be used to signal that an operation should be
    cancelled.
    
    Attributes:
        _cancelled (bool):
             A private attribute to keep track of the cancellation state. Initially False.
        _lock (Lock):
             A threading lock to ensure thread-safe modification of the cancellation state.
    
    Methods:
        __init__:
             Constructs a new `CancellationToken` instance and initializes the attributes.
        cancel:
             Sets the `_cancelled` attribute to True, signaling that a cancellation has been requested.
        cancelled:
             A property that returns the current cancellation state (True if cancelled, False otherwise).
        

    """

    def __init__(self):
        """
        Initializes the class instance by setting the initial state.
        This method performs the following actions:
        - Sets the `_cancelled` attribute to `False` indicating that the instance has not been cancelled.
        - Creates a new `Lock` instance and assigns it to the `_lock` attribute to control access to the resource in a multithreaded environment.
        

        """
        self._cancelled = False
        self._lock = Lock()

    def cancel(self) -> None:
        """
        Cancels the ongoing process represented by this instance.
        This method sets a flag that indicates that the process should be cancelled. It is thread-safe, as it
        acquires a lock before setting the cancellation flag.
        
        Returns:
            None

        """
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """
        Property that indicates whether the action has been cancelled or not.
        
        Returns:
            (bool):
                 True if the action has been cancelled, False otherwise.

        """
        return self._cancelled
