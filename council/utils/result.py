"""


module result
---------------

This module defines a Result type which is a union of Ok and Err classes, symbolizing successful and erroneous outcomes respectively. These classes offer a way to handle error states and normal states within operations in a consistent manner. Additionally, it provides a decorator function to wrap standard function calls in a try-except block, returning an instance of Ok with the function's result upon success, or an Err instance with the exception upon failure.

Classes:
    Ok: Represents a successful outcome holding an optional value.
    Err: Represents an error state holding an error information.

Functions:
    result_try(func: Callable) -> Callable:
        A decorator function that wraps the execution of 'func' in a try-except
        block. On successful execution of 'func', it returns an Ok instance
        containing the result. If an exception is raised, it catches the exception
        and returns an Err instance containing the exception.



"""
from typing import Union, Optional, Any


class Ok:
    """
    A class representing a successful outcome with an optional value.
    
    Attributes:
        value (Optional[Any]):
             The value contained by the Ok object. It can be of any type
            or None if no value is provided.
    
    Methods:
        __init__:
             Constructor for the Ok class.
        __repr__:
             Returns the official string representation of the Ok instance.
        is_ok:
             Static method indicating a successful outcome.
        is_err:
             Static method indicating the absence of an error (always False for Ok).

    """
    def __init__(self, value: Optional[Any] = None):
        """
        Initializes a new instance of the class with an optional value attribute.
        The constructor allows for an object to be created with an initial value, which can be of any type,
        or left as `None` by default if no value is provided.
        
        Args:
            value (Optional[Any], optional):
                 The initial value to assign to the instance. Defaults to None.
            

        """
        self.value = value

    def __repr__(self):
        """
        Representation method for the class instance.
        This magic method is used to obtain the official string representation
        of the class instance, which is particularly useful for debugging.
        It returns a string that would be used to recreate an object with
        the same value when passed to the eval() function.
        
        Returns:
            (str):
                 A string representing the object's value, formatted as 'Ok(<value>)'.
            

        """
        return f"Ok({self.value})"

    @staticmethod
    def is_ok() -> bool:
        """
        
        Returns a boolean indicating if the operation is okay or not.
            This method should be implemented by subclasses to perform
            a specific check. The default implementation always returns True.
        
        Returns:
            (bool):
                 Always true for the default implementation.

        """
        return True

    @staticmethod
    def is_err() -> bool:
        """
        
        Returns a boolean indicating whether an error has occurred or not.
            This method is designed to be used statically, to check at any point
            within a system if an error has been flagged. It consistently returns False,
            which implies that it's a placeholder or default implementation that
            indicates no error has been detected.
        
        Note that the method does not take any parameters and should not depend
            on any external state or instance variables since it is a static method.
        
        Returns:
            (bool):
                 False, indicating that no error has occurred.

        """
        return False


class Err:
    """
    A class representing an error state, containing methods to check whether an instance represents an error or a successful state.
    
    Attributes:
        error (Any):
             The stored error data or message.
    
    Methods:
        __init__(self, error):
            Initializes a new instance of the Err class with the provided error data.
        __repr__(self) -> str:
            Represents the instance as a string that includes the stored error.
        is_ok() -> bool:
            A static method that checks if the error state represents a successful state. Always returns False for the Err class.
        is_err() -> bool:
            A static method that checks if the error state represents an error. Always returns True for the Err class.

    """
    def __init__(self, error):
        """
        Initializes an instance of the class with an error message.
        This constructor stores the provided error message into a class member for later use.
        
        Args:
            error (str):
                 The error message to be stored in the instance.
            

        """
        self.error = error

    def __repr__(self):
        """
        
        Returns the official string representation of the object.
            This method computes the "official" string representation of an object,
            which is typically used for debugging. The representation is wrapped in an
            Err() object to signify that it represents an error state.
        
        Returns:
            (str):
                 A string representation of the error, enclosed in Err().

        """
        return f"Err({self.error})"

    @staticmethod
    def is_ok() -> bool:
        """
        
        Returns a boolean indicating whether a condition is ok or not.
            This static method always returns False, signifying that the condition it checks is not ok.
        
        Returns:
            (bool):
                 Always returns False.
            

        """
        return False

    @staticmethod
    def is_err() -> bool:
        """
        
        Returns a boolean indicating whether an error is present or not.
        
        Returns:
            (bool):
                 Always returns True, indicating the presence of an error.

        """
        return True


Result = Union[Ok, Err]


def result_try(func):
    """
    Wraps a callable such that it returns a `Result` type indicating success or failure.
    The `Result` type is typical in many programming languages to encapsulate a
    successful output or an error state without raising exceptions. When applied
    to a function, any result of the function is wrapped in an `Ok` result type
    signifying a successful outcome. If an exception occurs, it is caught and
    wrapped in an `Err` result type instead.
    
    Args:
        func (Callable):
             A callable to be executed within the `result_try` wrapper.
    
    Returns:
        (Callable):
             A wrapper function that captures exceptions and returns them
            as an `Err` result, or the result of the function call as an `Ok` result.
    
    Note:
        The `Result` type is not a built-in type in Python, and must be defined elsewhere
        within the application. Typically, `Result`, `Ok`, and `Err` would be part of a
        custom module or library designed to handle operations in a functional programming
        style where functions should not raise exceptions but return all possible outcomes
        explicitly.
        

    """
    def wrapper(*args, **kwargs) -> Result:
        """
        Wraps a function call in a 'Result' type pattern, returning 'Ok' with the result on success or 'Err' with the exception on failure.
        This function is designed to call any given function with positional and keyword arguments, which is useful for error handling. It catches any exception raised by the called function and returns it within an 'Err', otherwise it returns the result within an 'Ok'.
        
        Args:
            *args:
                 Variable length argument list for positional arguments to the wrapped function.
            **kwargs:
                 Arbitrary keyword arguments for the wrapped function.
        
        Returns:
            (Result):
                 An 'Ok' object with the result of the function call if it succeeds, or an 'Err' object with the exception information if it fails.

        """
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(e)

    return wrapper
