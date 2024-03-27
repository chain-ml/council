"""


Module llm_exception

This module provides custom exception classes for handling specific exceptions related to Large Language Model(LLM) operations.

Classes:
    LLMException (Exception): Base exception class for LLM related errors. It enriches the base Exception message
        with LLM-specific context by prepending 'llm:{llm_name}' if a language model name is provided.

    LLMCallTimeoutException (LLMException): Exception class that indicates a timeout has occurred during an LLM call.
        It takes an optional timeout value and includes it in the message alongside the llm_name if provided.

    LLMCallException (LLMException): Represents an exception that is raised when an LLM call encounters a non-success
        response code. It stores and exposes the received code and error through properties.

    LLMTokenLimitException (LLMException): Exception for scenarios where the number of tokens in a request exceeds the
        allowed limit for the specified model. It details the number of tokens, the limit, and the model in the message.

Attributes:
    None

Functions:
    None


"""
from typing import Optional


class LLMException(Exception):
    """
    A custom exception class that extends the built-in Exception class, specifically designed for errors related to Language Model (LLM) operations.
    
    Attributes:
        message (str):
             A human-readable message describing the exception.
        llm_name (Optional[str]):
             An optional string that denotes the name of the Language Model involved in the exception.
            The __init__ method enhances the base Exception message with the name of the Language Model if provided.
    
    Raises:
        TypeError:
             If `llm_name` is provided but is not a string.

    """

    def __init__(self, message: str, llm_name: Optional[str]) -> None:
        """
        Initializes a new instance of the class with a specified message and optionally a llm name.
        This constructor prepends the 'llm:' prefix to the llm name if a non-empty llm_name is provided,
        and then appends the message, otherwise, it uses the message as is. The modified or unmodified message
        is then passed to the superclass's constructor.
        
        Args:
            message (str):
                 The main message to be included in the exception.
            llm_name (Optional[str]):
                 The name of the llm, which will be prefixed to 'llm:' in the message.
                If the llm_name is None or an empty string, it will not be included in the message.
        
        Returns:
            (None):
                 This is a constructor method and thus does not return anything.
            

        """
        super().__init__(
            f"llm:{llm_name}, message {message}" if llm_name is not None and len(llm_name) > 0 else message
        )


class LLMCallTimeoutException(LLMException):
    """
    A custom exception that indicates a timeout occurred during a call to a language model (LLM).
    This exception is raised when a call to a LLM does not complete in the specified amount of time.
    It inherits from LLMException, which signifies it is a type of exception specifically related
    to language model operations.
    
    Attributes:
        timeout (Optional[float]):
             The duration in seconds after which the timeout occurred.
        llm_name (Optional[str]):
             The name of the language model that caused the exception.
    
    Args:
        timeout (Optional[float]):
             The maximum amount of time (in seconds) allotted for the LLM call,
            after which the exception is raised if the call does not complete.
        llm_name (Optional[str]):
             The name of the language model being called when the timeout happened.
    
    Raises:
        LLMException:
             An error indicating a timeout during a language model call.

    """

    def __init__(self, timeout: Optional[float], llm_name: Optional[str]) -> None:
        """
        Initializes an instance of the class with a timeout message and a name for the language model.
        
        Args:
            timeout (Optional[float]):
                 The number of seconds after which the LLM call will time out. If None, it indicates there is no timeout.
            llm_name (Optional[str]):
                 The optional name of the language model being used. If None, it means the model name is not specified.
        
        Raises:
            TypeError:
                 If initialization parameters do not match expected types.

        """
        super().__init__(f"LLM call timed out after {timeout} seconds", llm_name)


class LLMCallException(LLMException):
    """
    A custom exception that inherits from LLMException, meant for handling errors
    specific to LLM calls by including additional context such as the status code and
    error message from the LLM response.
    
    Attributes:
        _code (int):
             The status code indicating the nature of the error.
        _error (str):
             The error message providing details about the issue.
    
    Args:
        code (int):
             The status code returned by the LLM call that triggered the exception.
        error (str):
             A descriptive error message providing context for the exception.
        llm_name (Optional[str]):
             The name of the LLM that generated the error, passed to the superclass.
        Inherits:
        LLMException:
             The parent class that this exception is derived from.
        Properties:
        code (int):
             Getter property that returns the status code associated with the exception.
        error (str):
             Getter property that returns the error message associated with the exception.
        

    """

    def __init__(self, code: int, error: str, llm_name: Optional[str]) -> None:
        """
        Initializes the object with a specific error code and message.
        This constructor receives a status code, a descriptive error string, and an optional logical language model (llm) name, then constructs the instance by calling the superclass constructor with a formatted message integrating the code and error string. It also stores the code and error message in instance variables for later use.
        
        Args:
            code (int):
                 The error code representing the type of error encountered.
            error (str):
                 A detailed description of the error.
            llm_name (Optional[str]):
                 The name of the logical language model associated with the error, if any.
            

        """
        super().__init__(message=f"Wrong status code: {code}. Reason: {error}", llm_name=llm_name)
        self._code = code
        self._error = error

    @property
    def code(self) -> int:
        """
        Gets the code associated with this instance.
        This property is used to retrieve the private `_code` attribute which represents
        an integer code value.
        
        Returns:
            (int):
                 The integer code associated with this instance.

        """
        return self._code

    @property
    def error(self) -> str:
        """
        Property that gets the current error message.
        This read-only property returns the internally stored error message associated with an object instance.
        The error message is intended to provide a description of any exceptions or issues encountered during runtime.
        
        Returns:
            (str):
                 A string representing the current error message.
            

        """
        return self._error


class LLMTokenLimitException(LLMException):
    """
    Class to represent an exception for exceeding the token limit of a language model.
    This exception is raised when an operation attempts to process more tokens than the allowed limit
    for a specific language model. It extends from LLMException to provide tailored error messages
    for token limit issues.
    
    Attributes:
        token_count (int):
             The number of tokens that were attempted to be processed.
        limit (int):
             The maximum number of tokens allowed for the particular model.
        model (str):
             The name of the language model for which the limit has been exceeded.
        llm_name (Optional[str]):
             An optional name of the LLM operation or function during which
            the limit was exceeded, if applicable.
    
    Args:
        token_count (int):
             The number of tokens that were attempted to be processed.
        limit (int):
             The maximum number of tokens allowed for the particular model.
        model (str):
             The name of the language model for which the limit has been exceeded.
        llm_name (Optional[str]):
             Optionally, the name of the LLM operation or function during which
            the limit was exceeded.
        Inherits from:
        LLMException:
             A base exception class for language model-related errors.

    """

    def __init__(self, token_count: int, limit: int, model: str, llm_name: Optional[str]) -> None:
        """
        Initializes a new instance of the exception class with a specific message and optional large language model (LLM) name.
        This constructor is typically called when the token count exceeds the limit of a given model. It constructs an error message to indicate the infringement and initializes the base exception class with this message and an optional LLM name.
        
        Args:
            token_count (int):
                 The count of tokens that has exceeded the model's limit.
            limit (int):
                 The maximum number of tokens allowed for the model.
            model (str):
                 The name of the model for which the token limit was exceeded.
            llm_name (Optional[str]):
                 An optional name of the large language model involved in the error. If not provided, it defaults to None.
        
        Raises:
            This constructor does not explicitly raise exceptions; however, exceptions could be raised by the superclass initialization process.
            

        """
        super().__init__(f"token_count={token_count} is exceeding model {model} limit of {limit} tokens.", llm_name)
