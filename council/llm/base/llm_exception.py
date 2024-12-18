from typing import Optional, Sequence


class LLMException(Exception):
    """
    Custom exception for Large Language Model.
    """

    def __init__(self, message: str, llm_name: Optional[str]) -> None:
        """
        Initializes an instance of LLMException.

        Parameters:
            message (str): The error message
            llm_name (Optional[str]): The name of the LLM

        Returns:
            None
        """
        self.message = f"llm:{llm_name}, message {message}" if llm_name is not None and len(llm_name) > 0 else message
        super().__init__(self.message)


class LLMCallTimeoutException(LLMException):
    """
    Custom exception raised when a call to a Large Language Model timed out.
    """

    def __init__(self, timeout: Optional[float], llm_name: Optional[str]) -> None:
        """
        Initializes an instance of LLMCallException.

        Parameters:
            timeout (Optional[float]): The configured timeout
            llm_name (Optional[str]): The name of the LLM

        Returns:
            None
        """
        super().__init__(f"LLM call timed out after {timeout} seconds", llm_name)


class LLMCallException(LLMException):
    """
    Custom exception raised when the Large Language Model is executed.
    """

    def __init__(self, code: int, error: str, llm_name: Optional[str]) -> None:
        """
        Initializes an instance of LLMCallException.

        Parameters:
            code (int): The error code
            error (str): The error message
            llm_name (Optional[str]): The name of the LLM

        Returns:
            None
        """
        super().__init__(message=f"Wrong status code: {code}. Reason: {error}", llm_name=llm_name)
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

    def __init__(self, token_count: int, limit: int, model: str, llm_name: Optional[str]) -> None:
        """
        Initializes an instance of LLMTokenLimitException.

        Parameters:
            token_count (int): the actual number of tokens
            limit (int): the model limit
            model (str): the model
            llm_name Optional[str]: The name of the LLM
        Returns:
            None
        """
        super().__init__(f"token_count={token_count} is exceeding model {model} limit of {limit} tokens.", llm_name)


class LLMOutOfRetriesException(LLMException):
    """
    Custom exception raised when the maximum number of retries is reached.
    """

    def __init__(
        self, llm_name: Optional[str], retry_count: int, exceptions: Optional[Sequence[Exception]] = None
    ) -> None:
        """
        Initializes an instance of LLMOutOfRetriesException.
        """
        super().__init__(f"Exceeded maximum retries after {retry_count} attempts", llm_name)
        self.exceptions = exceptions if exceptions is not None else []
