"""

Module providing utilities to read and manage environment variables.

This module defines exceptions and functions specifically designed to
simplify the process of retrieving and handling environment variables within
your application, featuring type-specific retrieval with custom exception
handling.

Classes:
    MissingEnvVariableException: Exception raised when a required environment
    variable is missing.
    EnvVariableValueException: Exception raised when an environment variable
    does not conform to the expected type.

    OsEnviron: Context manager for temporarily setting environment variables.

Functions:
    read_env_str(name: str, required: bool = True, default: Optional[str] = None)
        -> Option[str]: Reads a string environment variable.

    read_env_int(name: str, required: bool = True, default: Optional[int] = None)
        -> Option[int]: Reads an integer environment variable.

    read_env_float(name: str, required: bool = True, default: Optional[float] = None)
        -> Option[float]: Reads a float environment variable.

    read_env_bool(name: str, required: bool = True, default: Optional[bool] = None)
        -> Option[bool]: Reads a boolean environment variable.

The '_read_env' function is a generic, internal function used by the
public read functions to fetch and convert environment variables.

Exceptions raised:
    MissingEnvVariableException: If a required environment variable is missing.
    EnvVariableValueException: If the value of an environment variable cannot
    be converted to the expected type.


"""
import os
from typing import Optional, Type, Callable, TypeVar, Any

from council.utils import Option


class MissingEnvVariableException(Exception):
    """
    A custom exception class for handling missing environment variables.
    This class is a specialized subclass of the built-in Exception. It's used to
    indicate that an expected environment variable is not found in the environment.
    This can be useful when environment variables are critical for the operation
    of the application and specific variables are required for the program to
    function correctly.
    
    Attributes:
        name (str):
             The name of the missing environment variable
    
    Args:
        name (str):
             A string representing the name of the missing environment variable.
    
    Raises:
        This class inherits all behaviors of the base Exception class and does not
        introduce any additional functionality upon instantiation or when raised.
        

    """

    def __init__(self, name: str):
        """
        Initializes a new instance of the class with a specified name for the missing environment variable.
        
        Args:
            name (str):
                 The name of the missing environment variable. This is used to customize the exception message.
        
        Raises:
            This constructor does not explicitly raise any exceptions, but invoking it may inherently raise an exception if the parent class's constructor does so.

        """
        self.name = name
        super().__init__(f"Missing required environment variable: {name}")


class EnvVariableValueException(Exception):
    """
    Exception raised when an environment variable has a value of an unexpected type.
    This exception is used to flag instances where the value of an environment variable does not match
    the expected type specified for it, for example, when a string is expected but a list is provided. It is
    a custom error that extends the Python Exception class.
    
    Attributes:
        name (str):
             The name of the environment variable with an invalid type.
        value (str):
             The actual value of the environment variable that caused the exception.
        expected_type (Type):
             The expected type of the environment variable's value.
        

    """

    def __init__(self, name: str, value: str, expected_type: Type):
        """
        Initialize an instance of the class with a name, value and expected type for an environment variable.
        This constructor is used to initialize an object when an environment variable has an invalid type.
        It sets the name of the variable and raises an exception with a detailed message specifying
        the name of the variable, its value, and the expected type.
        
        Args:
            name (str):
                 The name of the environment variable.
            value (str):
                 The current value of the environment variable.
            expected_type (Type):
                 The expected type for the value of the environment variable.
            

        """
        self.name = name
        super().__init__(f"Environment variable {name} value {value} has invalid type, expected: {expected_type}")


def read_env_str(name: str, required: bool = True, default: Optional[str] = None) -> Option[str]:
    """
    Reads an environment variable and returns it as a string.
    This function retrieves an environment variable and returns an
    Option wrapper containing the value as a string. If the environment
    variable is missing and is required, it raises a MissingEnvVariableException.
    If it is not required and has a default value provided, it returns a
    Option with the default value. If no default is provided, it returns
    an empty Option.
    
    Args:
        name (str):
             The name of the environment variable to retrieve.
        required (bool, optional):
             A flag indicating whether the environment variable
            is required. Defaults to True. If set to True and the environment
            variable is missing, an exception is raised.
        default (Optional[str], optional):
             The default value to return if the
            environment variable is missing and not required. Defaults to None.
    
    Returns:
        (Option[str]):
             An Option object containing the environment variable value
            as a string if it exists, default value if provided and not required,
            or an empty Option if neither is the case.
    
    Raises:
        MissingEnvVariableException:
             If the environment variable is required and missing.

    """
    return _read_env(name, required, default, lambda x: x)


def read_env_int(name: str, required: bool = True, default: Optional[int] = None) -> Option[int]:
    """
    Reads an environment variable and converts it to an integer.
    Reads the environment variable specified by `name` and converts its value to an integer. If the variable is not set and is required,
    an exception will be raised. If the variable is not set and is not required, the `default` value will be returned if provided,
    otherwise, an empty `Option` object will be returned. If a value is present but cannot be converted to an integer, an
    `EnvVariableValueException` will be raised.
    
    Args:
        name (str):
             The name of the environment variable to read.
        required (bool, optional):
             Whether the environment variable is required. Defaults to True.
        default (Optional[int], optional):
             The default value to return if the environment variable is not set. Defaults to None.
    
    Returns:
        (Option[int]):
             The converted value of the environment variable as an integer wrapped in an `Option`, or
            an empty `Option` if the variable is not set and a default value is not provided.
    
    Raises:
        MissingEnvVariableException:
             If the variable is required and not set in the environment.
        EnvVariableValueException:
             If the value cannot be converted to an integer.

    """
    def converter(x: str) -> int:
        """
        Converts a string to an integer with exception handling for value errors.
        This function attempts to convert a given string to an integer. If the conversion
        fails due to a ValueError, it raises an EnvVariableValueException, indicating
        that the provided string value is not valid for conversion to the expected type.
        
        Args:
            x (str):
                 The string value to convert into an integer.
        
        Returns:
            (int):
                 The converted integer.
        
        Raises:
            EnvVariableValueException:
                 If the conversion fails because the string cannot
                be converted to an integer.
        
        Note:
            This is an inner function that requires access to the 'name' variable defined
            in the outer scope, which represents the name of the environment variable being
            processed. The user-called functions like read_env_int handle the association
            between the 'name' and the 'converter' function.

        """
        try:
            return int(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, int) from e

    return _read_env(name, required, default, converter)


def read_env_float(name: str, required: bool = True, default: Optional[float] = None) -> Option[float]:
    """
    Reads a float value from an environment variable.
    This function attempts to retrieve the value of an environment variable specified
    by `name` and convert it to a float. If the environment variable is not set and
    a `default` value is provided, it returns an `Option` with that default. In case
    the variable is not set and no default is provided or if it's set but required,
    an appropriate Exception is raised.
    
    Args:
        name (str):
             The name of the environment variable to read.
        required (bool, optional):
             Indicates whether the environment variable is required. Defaults to True.
        default (Optional[float], optional):
             The default value to return if the environment variable is not set. Defaults to None.
    
    Returns:
        (Option[float]):
             An `Option` containing the retrieved and converted float value, or default if not set.
    
    Raises:
        EnvVariableValueException:
             If there's an error converting the environment variable's value to float.
        MissingEnvVariableException:
             If the required environment variable is not set.

    """
    def converter(x: str) -> float:
        """
        Converts a string to a float.
        This function attempts to convert a string to a float. If the conversion
        fails due to the string not being a valid representation of a float, it
        raises an EnvVariableValueException with the name of the environment variable,
        its invalid value, and the expected type (float).
        
        Args:
            x (str):
                 The string to convert to float.
        
        Returns:
            (float):
                 The converted float value of the string.
        
        Raises:
            EnvVariableValueException:
                 If the conversion fails because the string
                cannot be interpreted as a float, this exception is raised with
                relevant information.
            

        """
        try:
            return float(x)
        except ValueError as e:
            raise EnvVariableValueException(name, x, float) from e

    return _read_env(name, required, default, converter)


def read_env_bool(name: str, required: bool = True, default: Optional[bool] = None) -> Option[bool]:
    """
    Reads a boolean environment variable and returns its value as an `Option[bool]`.
    Tries to retrieve an environment variable with the given name and convert its value to a boolean. The conversion
    is case-insensitive and interprets 'true', '1', 't' as `True` and 'false', '0', 'f' as `False`. If the environment
    variable is not set and a default is provided, it returns the default encapsulated in an `Option`. If the
    environment variable is not set and no default is provided, it returns an `Option.none()`. If the environment
    variable is required but not set, it raises `MissingEnvVariableException`. If the value of the environment variable
    cannot be interpreted as a boolean, it raises `EnvVariableValueException`.
    
    Args:
        name (str):
             The name of the environment variable.
        required (bool, optional):
             If `True`, the variable is considered necessary and an exception is raised if
            it is missing. Defaults to `True`.
        default (Optional[bool], optional):
             The default value to return if the environment variable is not set.
            If `None`, no default value is provided. Defaults to `None`.
    
    Returns:
        (Option[bool]):
             An `Option` object containing the converted boolean value or the default value.
    
    Raises:
        MissingEnvVariableException:
             If the environment variable is required but not set.
        EnvVariableValueException:
             If the value cannot be converted to a boolean.
        

    """
    def converter(x: str) -> bool:
        """
        Converts a string to a boolean value.
        This function takes a string argument and returns a boolean value. It interprets common
        string representations of boolean values, such as 'true', '1', 't' for `True`, and 'false', '0', 'f' for `False`.
        If the string does not match any of the expected patterns, it raises an instance of `EnvVariableValueException`
        indicating that the value could not be converted to a boolean.
        
        Args:
            x (str):
                 The string to convert to a boolean.
        
        Returns:
            (bool):
                 The boolean value represented by the input string.
        
        Raises:
            EnvVariableValueException:
                 If the input cannot be converted to a boolean value,
                this exception is raised with a message indicating the name of the variable,
                the erroneous value, and the expected type (bool).

        """
        result = x.strip().lower()
        if result in ["true", "1", "t"]:
            return True
        if result in ["false", "0", "f"]:
            return False
        raise EnvVariableValueException(name, x, bool)

    return _read_env(name, required, default, converter)


T = TypeVar("T", covariant=True)


def _read_env(name: str, required: bool, default: Optional[T], convert: Callable[[str], T]) -> Option[T]:
    """
    Fetches the value of an environment variable and applies a conversion to it.
    This function retrieves an environment variable, then either converts it using
    a specified function or returns a default value if the environment variable is
    not found and not required. If the environment variable is not found, is required,
    and no default is given, it raises a `MissingEnvVariableException`.
    
    Args:
        name (str):
             The name of the environment variable to retrieve.
        required (bool):
             Specifies whether the environment variable is required.
        default (Optional[T]):
             The default value to return if the environment variable is not found.
        convert (Callable[[str], T]):
             A function to convert the environment variable string value to a desired type.
    
    Returns:
        (Option[T]):
             An `Option` containing the converted value of the environment variable if it exists,
            otherwise an `Option` containing the default value or an empty `Option` if no default is provided.
    
    Raises:
        MissingEnvVariableException:
             If the environment variable is required but not found.

    """
    result = os.getenv(name)
    if result is None:
        if required:
            raise MissingEnvVariableException(name)
        if default is not None:
            return Option.some(default)
        return Option.none()
    return Option.some(convert(result))


class OsEnviron:
    """
    A context manager to temporarily set the value of an environment variable within
    a block of code, restoring the original value upon exit of the block.
    
    Attributes:
        name (str):
             The name of the environment variable to modify.
        value (Optional[str]):
             The value to which the environment variable should be
            set. If set to None, the variable will be removed.
        previous_value (Optional[str]):
             The original value of the environment variable before
            modification that will be restored upon exit of the context manager block.
    
    Methods:
        __enter__:
             Called when entering the context of the `with` statement. It saves
            the current value of the environment variable and sets it to the new value.
        __exit__:
             Called when exiting the context of the `with` statement. It restores
            the environment variable to its original value.
        _set(value):
             A helper method to set or remove an environment variable.
            To use this class, it should be instantiated with the name and the desired temporary
            value of the environment variable, and then used within a `with` statement. Upon
            entering the `with` block, the environment variable will be set to the new value,
            and upon exit, it will be restored to its original value, ensuring that changes to
            the environment do not persist beyond the scope of the context manager.
        

    """
    def __init__(self, name: str, value: Optional[Any] = None):
        """
        Initialize a new instance of the class.
        This constructor sets up the initial state of the class instance by initializing the
        name and value attributes. It assigns the given name to the `name` attribute and
        converts the `value` to its string representation if it is not None; otherwise,
        it sets the `value` attribute to None. It also initializes `previous_value` with None
        indicating the absence of any previous value.
        
        Args:
            name (str):
                 The name to assign to the instance.
            value (Optional[Any], optional):
                 The initial value to associate with the instance.
                This value will be converted to a string. If not provided, the value will be set to None.
        
        Attributes:
            name (str):
                 The name assigned to the instance.
            value (Optional[str]):
                 The string representation of the `value` provided, or None if no value was given.
            previous_value (Optional[str]):
                 A placeholder for storing the previous value, initialized to None.
            

        """
        self.name = name
        self.value = str(value) if value is not None else None
        self.previous_value = None

    def __enter__(self):
        """
        Handles the entry into a runtime context related to an environment variable.
        Upon entering a context managed by this object using the 'with' statement, it stores the previous value of the environment variable
        with the given 'name' and sets it to the new 'value'. The environment variable will be restored or cleared when exiting
        the context.
        
        Attributes:
            previous_value:
                 The value of the environment variable before entering the context.
        
        Note:
            This function is meant to be used in conjunction with the Python 'with' statement and the __exit__ method.
            

        """
        self.previous_value = os.environ.get(self.name, None)
        self._set(self.value)

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Exits a runtime context related to this object. This function is designed to be used in conjunction with the `with` statement which allows for setup and teardown code for resources. In a class that implements context management protocol, it is called when execution leaves the context of the `with` statement. Typically used for cleaning up resources or restoring the state that was modified within the `with` context. This method ensures that the previous value is reset to its original state, by calling `_set` method with the `previous_value` attribute when the `with` block is exited, regardless of whether an exception was raised or not within the `with` block.
        
        Args:
            exception_type (type):
                 Type of the exception raised while executing the block wrapped by the `with` statement, if any. None if no exception occurred.
            exception_value (Exception):
                 The exception instance raised. None if no exception occurred.
            traceback (TracebackType):
                 The traceback object representing the traceback at the point where the exception was raised. None if no exception occurred.
        
        Returns:
            (None):
                 This method does not return any value but it is required to correctly implement the context management protocol.
        
        Raises:
            This method should not explicitly raise exceptions but any exceptions raised by the `_set` method will be propagated.

        """
        self._set(self.previous_value)

    def _set(self, value: Optional[str]):
        """
        Sets the environment variable to the given value, or removes it if the value is None.

        """
        os.environ.pop(self.name, None)
        if value is not None:
            os.environ[self.name] = value

    def __str__(self):
        """
        Generate a string representation of the environment variable object.
        This method returns a string that includes the environment variable's name, current value, and its previous value if it has been changed.
        
        Returns:
            (str):
                 A formatted string containing the name and current value of the environment variable, along with its previous value.

        """
        return f"Env var:`{self.name}` value:{self.value} (previous value: {self.previous_value})"
