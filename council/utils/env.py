import os
from typing import Optional


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


def read_env(name: str, required: bool = True, default: Optional[str] = None) -> str:
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return default
        return ""
    return result
