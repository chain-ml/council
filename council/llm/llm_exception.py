from typing import Optional


class LLMException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class LLMCallTimeoutException(LLMException):
    def __init__(self, timeout: Optional[float]):
        super().__init__(f"Call to LLM timed out after {timeout} seconds")


class LLMCallException(LLMException):
    """
    Custom exception raised when the Large Language mModel is executed.
    """

    def __init__(self, code: int, error: str):
        super().__init__(f"Wrong status code: {code}. Reason: {error}")
        self._code = code
        self._error = error

    @property
    def code(self) -> int:
        return self._code

    @property
    def error(self) -> str:
        return self._error


class LLMTokenLimitException(LLMException):
    """
    Custom exception raised when the number of tokens exceed the model limit.
    """

    def __init__(self, token_count: int, limit: int, model: str):
        """
        Initializes an instance of LLMTokenLimitException.

        Parameters:
            token_count (int): the actual number of tokens
            limit (int): the model limit
            model (str): the model

        Returns:
            None
        """
        super().__init__(f"token_count={token_count} is exceeding model {model} limit of {limit} tokens.")
