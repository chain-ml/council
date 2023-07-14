import os
from typing import Optional, Type

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


class InvalidTypeEnvVariableException(Exception):
    """
    Custom exception raised when an environment variable type is incorrect.
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
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()
    return Option.some(result)


def read_env_int(name: str, required: bool = True, default: Optional[int] = None) -> Option[int]:
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()

    try:
        return Option.some(int(result))
    except ValueError:
        raise InvalidTypeEnvVariableException(name, result, int)


def read_env_float(name: str, required: bool = True, default: Optional[float] = None) -> Option[float]:
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()

    try:
        return Option.some(float(result))
    except ValueError:
        raise InvalidTypeEnvVariableException(name, result, float)


def read_env_bool(name: str, required: bool = True, default: Optional[bool] = None) -> Option[bool]:
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()

    if result.lower() in ['true', '1', 't']:
        return Option.some(True)
    if result.lower() in ['false', '0', 'f']:
        return Option.some(False)
    raise InvalidTypeEnvVariableException(name, result, bool)
