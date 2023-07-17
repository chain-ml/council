import os
from typing import Optional, Type, Callable, TypeVar

from council.utils import Option


class MissingEnvVariableException(Exception):
    """
    Custom exception raised when a required environment variable is missing.
    """

    def __init__(self, name: str):
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

    def __init__(self, name: str, value: str, expected_type: Type):
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
    return _read_env(name, required, default, lambda x: x)


def read_env_int(name: str, required: bool = True, default: Optional[int] = None) -> Option[int]:
    def converter(x: str) -> int:
        try:
            return int(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, int) from e

    return _read_env(name, required, default, converter)


def read_env_float(name: str, required: bool = True, default: Optional[float] = None) -> Option[float]:
    def converter(x: str) -> float:
        try:
            return float(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, float) from e

    return _read_env(name, required, default, converter)


def read_env_bool(name: str, required: bool = True, default: Optional[bool] = None) -> Option[bool]:
    def converter(x: str) -> bool:
        result = x.strip().lower()
        if result in ["true", "1", "t"]:
            return True
        if result in ["false", "0", "f"]:
            return False
        raise EnvVariableValueException(name, x, bool)

    return _read_env(name, required, default, converter)


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
