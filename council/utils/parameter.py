"""

Module that defines custom parameter validation, conversion, and management mechanisms.

This module provides tools for defining parameters with custom validation, type conversion from environment variables, and managing parameter values with optional default values or explicit requirements for user provision.

Classes:
    Parameter(Generic[T]) - A generic parameter class for managing parameter values and validation.
    ParameterValueException(Exception) - Exception raised when an invalid parameter value is encountered.

Functions:
    greater_than_validator(value: int) -> Validator - Returns a validator function that checks if an int value is greater than a specified value.
    prefix_validator(value: str) -> Validator - Returns a validator function that checks if a string value starts with a specified prefix.
    not_empty_validator(x: str) - Validates that a given string is not empty or whitespace.

Types:
    Validator - Type alias for a Callable that takes a value of type T and performs validation, raising ValueError on failure.
    OptionalOrUndefined - Type representing a value that is either of optional type T or an instance of Undefined.
    _Converter - Private type alias for a conversion function that converts a string value to type T and takes a flag to indicate requirement.

Variables:
    T - A type variable used for generic typing.
    _undefined - A module-level constant instance of Undefined, representing the absence of a value.



"""
from __future__ import annotations

from typing import Callable, TypeVar, Optional, Generic, Any, Union

from council.utils import Option, read_env_int, read_env_float, read_env_str

T = TypeVar("T")
Validator = Callable[[T], None]
_Converter = Callable[[str, bool], T]


def greater_than_validator(value: int) -> Validator:
    """
    Creates a validator function that will check if the input integer is greater than a specified value.
    The `greater_than_validator` function takes an integer `value` and returns a `validator` function. The returned `validator` function then takes an integer `x` as its input and checks if `x` is greater than the `value` provided when the `greater_than_validator` was called. If `x` is not greater than `value`, it raises a `ValueError` with a message indicating the requirement.
    
    Args:
        value (int):
             An integer to which the input `x` will be compared.
    
    Returns:
        A `validator` function which can be called with an integer argument.
        

    """
    def validator(x: int):
        """
        Validates that a given integer `x` is greater than a predefined value.
        This function checks if the input integer `x` is strictly greater than a global `value`. If `x` is less than or equal to `value`,
        a ValueError is raised with an appropriate error message.
        
        Args:
            x (int):
                 The integer to validate.
        
        Raises:
            ValueError:
                 If `x` is less than or equal to the predefined `value`.
            

        """
        if x <= value:
            raise ValueError(f"must be greater than {value}")

    return validator


def prefix_validator(value: str) -> Validator:
    """
    Validates whether the strings provided to the returned validator function start with a specified prefix.
    This function acts as a factory that generates a nested validator function. The generated function takes a single string argument and checks if it starts with the prefix provided to `prefix_validator`. If the string does not start with the specified prefix, the validator function raises a `ValueError`. To use this validator, call `prefix_validator` with the desired prefix, and then use the returned function to validate strings.
    
    Args:
        value (str):
             The prefix that the input strings must start with.
    
    Returns:
        (Callable[[str], None]):
             A validator function that takes a single string argument and raises a `ValueError` if the argument does not start with the specified prefix.
        

    """
    def validator(x: str):
        """
        Validator for checking if a provided string starts with a predetermined prefix.
        This function checks if the given string `x` begins with a specific prefix defined by the
        variable `value`. If `x` does not start with the required prefix, the function raises
        a `ValueError` exception with an appropriate error message.
        
        Args:
            x (str):
                 The string to validate against the predefined prefix.
        
        Raises:
            ValueError:
                 If `x` does not start with the prefix defined in `value`.
            

        """
        if not x.startswith(value):
            raise ValueError(f"must start with `{value}`")

    return validator


def not_empty_validator(x: str):
    """
    Checks if the input string is not empty after stripping whitespace.
    
    Args:
        x (str):
             The string to validate.
    
    Raises:
        ValueError:
             If `x` is empty or contains only whitespace.

    """
    if len(x.strip()) == 0:
        raise ValueError("must not be empty")


class Undefined:
    """
    A placeholder class that represents an undefined value within a given context.
    This class does not carry any functionality beyond being a representation of an absence of a more specific value. It mainly serves as a marker or a sentinel object. It has a single method that overrides the default string representation to return the string 'Undefined'.
    
    Attributes:
        None
    
    Methods:
        __repr__:
             Returns a string that simply states 'Undefined'. Useful for debugging and logging purposes to signal that a value is purposefully undefined.
        

    """

    def __repr__(self) -> str:
        """
        A special method used to get the official string representation of the object, which can be helpful for debugging.
        This method has been intentionally left vague and simply returns the string 'Undefined'.
        Subclasses should override this method to provide a more descriptive representation
        that is helpful during debugging and logging.
        
        Returns:
            (str):
                 A string that represents the object. Currently, this is hardcoded to 'Undefined'.

        """
        return "Undefined"


OptionalOrUndefined = Union[Optional[T], Undefined]

_undefined = Undefined()


class ParameterValueException(Exception):
    """
    A custom exception for signaling that a parameter value is invalid.
    This exception is raised when a function argument or parameter does not meet the expected criteria. It is a subclass of the base `Exception` class and provides a more descriptive error message that includes the name of the parameter, the invalid value provided, and an additional message explaining what the valid criteria are.
    
    Attributes:
        name (str):
             The name of the parameter that has an invalid value.
        value (Any):
             The invalid value that was provided for the parameter.
        message (Exception):
             An exception instance containing information about what the valid criteria for the parameter value are.

    """

    def __init__(self, name: str, value: Any, message: Exception):
        """
        Initializes an exception object with a specific error message that includes the invalid parameter name, value, and a custom message.
        This exception should be raised when a function's parameter value does not meet expected criteria. It constructs a detailed error message indicating the name of the parameter, the incorrect value, and a supplementary message provided during instantiation.
        
        Args:
            name (str):
                 The name of the parameter that received an invalid value.
            value (Any):
                 The actual value of the parameter that was deemed invalid.
            message (Exception):
                 An exception object or message that indicates the criteria for a valid value.
        
        Raises:
            This constructor does not explicitly raise exceptions but calling it incorrectly could lead to a TypeError or ValueError if the provided arguments do not match the expected types or format.
            

        """
        super().__init__(f"'{name}' parameter value '{value}' is invalid. Value must be {message}")


class Parameter(Generic[T]):
    """
    A generic class to handle parameters of various types with optional validation and default values.
    This class is designed to handle different types of parameters that are required or optional,
    with the capability to automatically convert, validate, and retrieve their values, including
    from environment variables. It provides a structured way to declare parameters with
    additional metadata and behavior, such as default values and validation functions.
    
    Attributes:
        _name (str):
             The name of the parameter.
        _required (bool):
             Indicates whether the parameter is required.
        _validator (Validator):
             Validation function to validate the parameter value. Defaults to a no-op if None is provided.
        _default (OptionalOrUndefined[T]):
             Default value of the parameter, can be undefined.
        _value (Option[T]):
             The actual value of the parameter, can be none if not set or undefined.
        _read_env (_Converter):
             Function to read and convert environment variable to parameter type.
    
    Args:
        name (str):
             The name of the parameter.
        required (bool):
             Indicates whether the parameter is required.
        converter (_Converter):
             Function to read and convert environment variable to parameter type.
        value (OptionalOrUndefined[T], optional):
             Initial value of the parameter.
        default (OptionalOrUndefined[T], optional):
             Default value of the parameter.
        validator (Optional[Validator], optional):
             Function that validates the parameter's value. Defaults to None.
    
    Raises:
        ParameterValueException:
             If validation fails upon setting the parameter's value.
        

    """
    def __init__(
        self,
        name: str,
        required: bool,
        converter: _Converter,
        value: OptionalOrUndefined[T] = _undefined,
        default: OptionalOrUndefined[T] = _undefined,
        validator: Optional[Validator] = None,
    ):
        """
        Initializes a new instance of the object with the specified parameters.
        
        Args:
            name (str):
                 The unique identifier for the object.
            required (bool):
                 A flag indicating whether the object requires a value.
            converter (_Converter):
                 A type or function that converts the input value to the desired format.
            value (OptionalOrUndefined[T], optional):
                 The initial value provided for the object. If undefined, the default will be used. Defaults to _undefined.
            default (OptionalOrUndefined[T], optional):
                 The default value for the object if no initial value is provided. Defaults to _undefined.
            validator (Optional[Validator], optional):
                 A function that checks if the value meets certain criteria. Defaults to None, which sets a no-op validator.
                This method uses the supplied converter to process the provided value or the default, sets up a validator if given, and stores the information required to identify and define the value's characteristics within the object.

        """
        self._name = name
        self._required = required
        self._validator: Validator = validator if validator is not None else lambda x: None
        self._default = default
        if isinstance(value, Undefined):
            if not isinstance(default, Undefined):
                self.set(default)
            else:
                self._value: Option[T] = Option.none()
        else:
            self.set(value)

        self._read_env = converter

    def from_env(self, env_var: str) -> None:
        """
        Pulls a value from an environment variable and updates the instance's setting with that value.
        This method attempts to read the value of an environment variable specified by `env_var` and, if it finds a value,
        it calls the `set` method to update the setting. If the environment variable is not set and the setting is required,
        it will raise an appropriate exception.
        
        Args:
            env_var (str):
                 The name of the environment variable to read.
        
        Raises:
            An exception if the environment variable is required but not set.
        
        Returns:
            None

        """
        v = self._read_env(env_var, self._required)
        if v.is_some():
            self.set(v.unwrap())

    def set(self, value: Optional[T]) -> None:
        """
        Set the value of an option after running a validation check.
        This function sets a value to an underlying Option object after performing a validation
        check through a provided validator function. If the validation fails, a ParameterValueException
        is raised with details about the invalid value and the requirements.
        
        Args:
            value (Optional[T]):
                 The value to set. The type `T` represents the type parameter
                for the Option object.
        
        Raises:
            ParameterValueException:
                 If the provided value fails the validation check,
                this exception is raised with details about the name of the parameter, the invalid
                value provided, and a message indicating the validation requirements.
        
        Returns:
            (None):
                 This method doesn't return anything.
            

        """
        try:
            self._validator(value)
            self._value = Option(value)
        except ValueError as e:
            raise ParameterValueException(self._name, value=value, message=e)

    @property
    def name(self) -> str:
        """
        Gets the name of an object.
        This property method is used to retrieve the private _name attribute from
        an object. Since it's decorated with @property, it can be accessed as an
        attribute without the need to call it as a method.
        
        Returns:
            (str):
                 The name of the object.

        """
        return self._name

    @property
    def value(self) -> Optional[T]:
        """
        Property that retrieves the wrapped value of an optional-like object.
        This property accesses the internal value of the object, unwrapping it if it is present (i.e., `is_some()` method returns `True`). If the value is not present (when `is_some()` returns `False`), then it returns `None`.
        
        Returns:
            (Optional[T]):
                 The unwrapped value if present, otherwise `None`.
            

        """
        return self._value.unwrap() if self.is_some() else None

    @property
    def required(self) -> bool:
        """
        Property that indicates whether a parameter is required or not.
        The property returns True if the parameter is required, otherwise False.
        
        Returns:
            (bool):
                 A boolean value indicating the requirement status of a parameter.
            

        """
        return self._required

    @property
    def is_default(self) -> bool:
        """
        Property that determines if the current value equals the default value.
        This property checks whether the current value is equivalent to a pre-defined default value. It primarily compares
        the wrapped value of the object against the default, but also incorporates a check to handle cases where
        the value is not defined (`Undefined`).
        
        Returns:
            (bool):
                 Returns True if the current value is the same as the default value; otherwise, returns False.
                This includes returning False if the current value is considered 'None' or undefined.
            

        """
        if isinstance(self._default, Undefined):
            return False
        if self._value.is_none():
            return False
        return self._value.unwrap() == self._default

    def unwrap(self) -> T:
        """
        Unwraps the contained value within an object instance. The method assumes that the instance has an internal attribute `_value` that is wrapped and can be retrieved. It is expected that the wrapped value is of a specific type `T`, which should be returned by this method.
        
        Returns:
            (T):
                 The unwrapped value of the inner `_value` attribute.
        
        Raises:
            AttributeError:
                 If the `_value` attribute does not exist or cannot be unwrapped.
                Any other exception that might be raised during the unwrapping process depending on the implementation of `unwrap` method.

        """
        return self._value.unwrap()

    def unwrap_or(self, value: Any) -> Union[T, Any]:
        """
        
        Returns the contained value or a default.
        
        Args:
            value (Any):
                 The default value to return if the option is None.
        
        Returns:
            (Union[T, Any]):
                 The contained value if the option is Some; otherwise, returns the provided default value.
        
        Raises:
            AttributeError:
                 If the method 'unwrap_or' is called on an object that does not have the '_value' attribute with an 'unwrap_or' method.

        """
        return self._value.unwrap_or(value)

    def is_some(self) -> bool:
        """
        Checks if the option has some value or not.
        
        Returns:
            (bool):
                 True if the option has some value, False otherwise.

        """
        return self._value.is_some()

    def is_none(self) -> bool:
        """
        Checks if the attribute '_value' is None.
        This method determines whether the internal attribute '_value' of the class instance
        is set to None, indicating the absence of a value.
        
        Returns:
            (bool):
                 True if '_value' is None, otherwise False.
            

        """
        return self._value.is_none()

    def __str__(self) -> str:
        """
        
        Returns a string representation of the parameter including its optional status, value,
            and default value if defined.
            This method constructs a human-readable representation of the parameter instance
            by considering its required status, current value, and default value. The outcome of the method
            (is a descriptive string that includes):
                - The word '(optional)' if the parameter is not required.
                - The current value of the parameter if it is set, otherwise 'undefined value'.
                - The default value of the parameter if it has one that is not an instance of Undefined.
        
        Returns:
            (str):
                 The string representation of the parameter which includes details on its optional status,
                current value, and default value if present.
            

        """
        opt = "(optional)" if not self._required else ""
        val = "undefined value" if self.is_none() else f"value `{self._value.unwrap()}`"
        default = f" Default value `{self._default}`." if not isinstance(self._default, Undefined) else ""
        return f"Parameter{opt} `{self._name}` with {val}.{default}"

    def __eq__(self, other: Any) -> bool:
        """
        Check if the current Parameter object is equal to another object.
        This method overrides the equality operator (==) to compare the current object either with another Parameter object or any other object that can be compared with the content wrapped by this Parameter. It first checks if the current object is None (by means of `is_none` method). If it is, the method checks whether the other object is also a Parameter object that is None. If the current object is not None, it then checks if the other object is a Parameter object and proceeds to compare their underlying values (unwrapped using `unwrap` method). If the other object is not a Parameter object, it compares the unwrapped value of the current object directly with the other object.
        
        Args:
            other (Any):
                 The object to compare with the current Parameter object.
        
        Returns:
            (bool):
                 True if both objects are considered equal, otherwise False.
            

        """
        if self.is_none():
            if isinstance(other, Parameter):
                return other.is_none()
            return False

        if isinstance(other, Parameter):
            return self.unwrap() == other.unwrap()
        return self.unwrap() == other

    @staticmethod
    def string(
        name: str,
        required: bool,
        value: OptionalOrUndefined[str] = _undefined,
        default: OptionalOrUndefined[str] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[str]:
        """
        Create a string-typed `Parameter` instance with optional validation and defaults.
        This method acts as a factory for creating `Parameter` objects that are specifically meant to hold string
        values. It allows for setting a default value, validation, and specifying whether the parameter is required.
        
        Args:
            name (str):
                 The name identifier for the parameter.
            required (bool):
                 Flag indicating whether the parameter is required.
            value (OptionalOrUndefined[str], optional):
                 The initial value to set for the parameter. If undefined,
                the default value will be used if provided. Defaults to _undefined.
            default (OptionalOrUndefined[str], optional):
                 The default value for the parameter if no value is
                specified. Defaults to _undefined.
            validator (Optional[Validator], optional):
                 A function to validate the parameter value. The function should
                raise a `ValueError` if validation fails. Defaults to None.
        
        Returns:
            (Parameter[str]):
                 A string-typed `Parameter` object initialized with the provided arguments.
            

        """
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_str,
            default=default,
            validator=validator,
        )

    @staticmethod
    def int(
        name: str,
        required: bool,
        value: OptionalOrUndefined[int] = _undefined,
        default: OptionalOrUndefined[int] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[int]:
        """
        Creates a new `Parameter` instance for integer values.
        This factory method simplifies the creation of `Parameter` objects that specifically handle integer values. It utilizes the `read_env_int` function to convert environment variable strings into integers and applies any specified validator to ensure the value adheres to given constraints.
        
        Args:
            name (str):
                 The name of the parameter.
            required (bool):
                 A flag indicating whether the parameter is required or optional.
            value (OptionalOrUndefined[int], optional):
                 The initial value of the parameter if provided. Defaults to `_undefined`.
            default (OptionalOrUndefined[int], optional):
                 The default value of the parameter if not explicitly set. Defaults to `_undefined`.
            validator (Optional[Validator], optional):
                 A function to validate the value of the parameter. Defaults to None.
        
        Returns:
            (Parameter[int]):
                 An instance of `Parameter` typed to handle integers.

        """
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_int,
            default=default,
            validator=validator,
        )

    @staticmethod
    def float(
        name: str,
        required: bool,
        value: OptionalOrUndefined[float] = _undefined,
        default: OptionalOrUndefined[float] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[float]:
        """
        Creates a new Parameter instance with type annotation `float`.
        This method acts as a factory function to create a `Parameter[float]` instance, specifying the behavior and constraints for a
        parameter that should receive a floating-point number. It leverages a function to read environment variables and convert them
        into a float, and applies optional validation.
        
        Args:
            name (str):
                 The name of the parameter.
            required (bool):
                 A flag indicating whether the parameter is required.
            value (OptionalOrUndefined[float], optional):
                 The initial value of the parameter, if provided. Defaults to `_undefined`.
            default (OptionalOrUndefined[float], optional):
                 The default value of the parameter, if the initial value is not provided.
                Defaults to `_undefined`.
            validator (Optional[Validator], optional):
                 An optional function to validate the parameter's value. Defaults to `None`.
        
        Returns:
            (Parameter[float]):
                 An instance of `Parameter` with the specified configurations.
            

        """
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_float,
            default=default,
            validator=validator,
        )
