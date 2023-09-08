from typing import Generic, TypeVar

T = TypeVar("T")


class Monitored(Generic[T]):
    def __init__(self, name: str, monitorable: T):
        self._name = name
        self._inner = monitorable

    @property
    def inner(self) -> T:
        return self._inner

    @property
    def name(self) -> str:
        return self._name
