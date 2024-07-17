from threading import Lock


class CancellationToken:
    """
    A cancellation token which is initially not set.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._lock = Lock()

    def cancel(self) -> None:
        """
        set the cancellation token.
        """
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """
        returns `True` if the cancellation token is set, otherwise, `False`
        """
        return self._cancelled
