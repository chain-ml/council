from typing import Union, Optional, Any


class Ok:
    def __init__(self, value: Optional[Any] = None):
        self.value = value

    def __repr__(self):
        return f"Ok({self.value})"

    @staticmethod
    def is_ok() -> bool:
        return True

    @staticmethod
    def is_err() -> bool:
        return False


class Err:
    def __init__(self, error):
        self.error = error

    def __repr__(self):
        return f"Err({self.error})"

    @staticmethod
    def is_ok() -> bool:
        return False

    @staticmethod
    def is_err() -> bool:
        return True


Result = Union[Ok, Err]


def result_try(func):
    def wrapper(*args, **kwargs) -> Result:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(e)

    return wrapper
