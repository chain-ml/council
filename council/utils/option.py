from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class OptionException(Exception):
    pass


class Option(Generic[T]):
    """
    Convenient class to manage optional values.
    """

    def __init__(self, some: Optional[T]) -> None:
        """
        Initialize a new instance

        Parameters:
            some (Optional[T]): some value, if any. Otherwise, `None`
        """
        self._some = some

    def unwrap(self, message: Optional[str] = None) -> T:
        """
        unwrap the value in the instance.

        Parameters:
            message (Optional(str)): error message to be set on the :class:`~.OptionException`
            if there is no value to unwrap
        returns:
            T: the value

        raises:
            OptionException: there is no value wrapped by this instance
        """
        if self._some is not None:
            return self._some

        raise OptionException(message if message is not None else "")

    def unwrap_or(self, default: T) -> T:
        """
        returns the wrap value if some, or the provided default

        Parameters:
            default(T): default value

        Returns:
            T:
        """
        if self._some is not None:
            return self._some
        return default

    def map_or(self, map_func: Callable[[T], R], default: R) -> R:
        """
        returns the result of the give map function on wrap value if some, or the provided default

        Parameters:
            map_func(Callable[[T], R]): map function applies on the value if some
            default(R): default value

        Returns:
            R:
        """

        if self._some is not None:
            return map_func(self._some)
        return default

    def as_optional(self) -> Optional[T]:
        """
        Returns this instance as an optional
        """
        return self._some

    def is_none(self) -> bool:
        """
        Returns `True` is this instance does not contain any value
        """
        return self._some is None

    def is_some(self) -> bool:
        """
        Returns `True` is this instance contains some value
        """
        return not self.is_none()

    @staticmethod
    def some(some: T) -> Option[T]:
        """
        Create a new instance with some value.

        Parameters:
            some (T): the value to be wrapped by this instance
        Returns:
            Option[T]: a new instance
        """
        return Option(some)

    @staticmethod
    def none() -> Option[T]:
        """
        Create a new instance with none

        Returns:
              Option[T]: a new instance
        """
        return Option(None)

    def __repr__(self) -> str:
        return "Option(None)" if self.is_none() else f"Option({self._some})"

    def __str__(self) -> str:
        return "none" if self.is_none() else f"{self._some}"
