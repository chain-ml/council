import os
from typing import Any, Callable, Optional, Type, TypeVar

from council.utils import Option


class MissingEnvVariableException(Exception):
    """
    Custom exception raised when a required environment variable is missing.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes an instance of MissingEnvVariableError.

        Parameters:
            name (str): The name of the missing environment variable.

        Returns:
            None
        """
        self.name = name
        super().__init__(f"Missing required environment variable: {name}")


class EnvVariableValueException(Exception):
    """
    Custom exception raised if an environment variable is assigned a value
    that is inconsistent with its declared data type.
    """

    def __init__(self, name: str, value: str, expected_type: Type) -> None:
        """
        Initializes an instance of MissingEnvVariableError.

        Parameters:
            name (str): The name of the environment variable.

        Returns:
            None
        """
        self.name = name
        super().__init__(f"Environment variable {name} value {value} has invalid type, expected: {expected_type}")


def read_env_str(name: str, required: bool = True, default: Optional[str] = None) -> Option[str]:
    """Read an environment variable as string, return as Option."""
    return _read_env(name, required, default, lambda x: x)


def must_read_env_str(name: str) -> str:
    """Read an environment variable as string."""
    return read_env_str(name, required=True).unwrap()


def read_env_int(name: str, required: bool = True, default: Optional[int] = None) -> Option[int]:
    """Read an environment variable as integer, return as Option."""

    def converter(x: str) -> int:
        try:
            return int(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, int) from e

    return _read_env(name, required, default, converter)


def must_read_env_int(name: str) -> int:
    """Read an environment variable as integer."""
    return read_env_int(name, required=True).unwrap()


def read_env_float(name: str, required: bool = True, default: Optional[float] = None) -> Option[float]:
    """Read an environment variable as float, return as Option."""

    def converter(x: str) -> float:
        try:
            return float(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, float) from e

    return _read_env(name, required, default, converter)


def must_read_env_float(name: str) -> float:
    """Read an environment variable as float."""
    return read_env_float(name, required=True).unwrap()


def read_env_bool(name: str, required: bool = True, default: Optional[bool] = None) -> Option[bool]:
    """Read an environment variable as boolean, return as Option."""

    def converter(x: str) -> bool:
        result = x.strip().lower()
        if result in ["true", "1", "t"]:
            return True
        if result in ["false", "0", "f"]:
            return False
        raise EnvVariableValueException(name, x, bool)

    return _read_env(name, required, default, converter)


def must_read_env_bool(name: str) -> bool:
    """Read an environment variable as boolean."""
    return read_env_bool(name, required=True).unwrap()


T = TypeVar("T", covariant=True)


def _read_env(name: str, required: bool, default: Optional[T], convert: Callable[[str], T]) -> Option[T]:
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()
    return Option.some(convert(result))


class OsEnviron:
    def __init__(self, name: str, value: Optional[Any] = None):
        self.name = name
        self.value = str(value) if value is not None else None
        self.previous_value = None

    def __enter__(self):
        self.previous_value = os.environ.get(self.name, None)
        self._set(self.value)

    def __exit__(self, exception_type, exception_value, traceback):
        self._set(self.previous_value)

    def _set(self, value: Optional[str]):
        os.environ.pop(self.name, None)
        if value is not None:
            os.environ[self.name] = value

    def __str__(self) -> str:
        return f"Env var:`{self.name}` value:{self.value} (previous value: {self.previous_value})"
