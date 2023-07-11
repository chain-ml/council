from threading import Lock


class CancellationToken:
    def __init__(self):
        self._cancelled = False
        self._lock = Lock()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled
