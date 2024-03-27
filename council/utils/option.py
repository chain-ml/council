"""


Module option

This module provides a generic Option type for conditional handling of values without explicitly using if statements for checks. It offers a way to avoid problems that might come from handling potentially None (null) variables, which is commonly referred to as None-safe or null-safe programming.

Classes:
    OptionException - Custom exception for handling no value cases.
    Option(Generic[T]) - Represents an encapsulation of an optional value.

Typevar:
    T - The type of the value that can be contained in an Option.
    R - The return type used in map_or method.

Details:
The module contains the Option class that provides methods for interacting with optional values:

- __init__(self, some: Optional[T])
  Constructor that takes a value of type Optional[T].

- unwrap(self, message: Optional[str]=None) -> T
  Retrieves the contained value or raises an OptionException.

- unwrap_or(self, default: T) -> T
  Retrieves the contained value or provides a default if the value is None.

- map_or(self, map_func: Callable[[T], R], default: R) -> R
  Applies a function to the contained value if it's not None, or returns a default value.

- as_optional(self) -> Optional[T]
  Returns the contained value as an Optional type.

- is_none(self) -> bool
  Checks whether the Option is None.

- is_some(self) -> bool
  Checks whether the Option contains a value.

- Option.some(some: T) -> Option[T]
  Class method to create an Option with a given value.

- Option.none() -> Option[T]
  Class method to create an Option with no value.

Methods like __repr__ and __str__ are also defined for better readability and string representation of the Option objects.


"""
from __future__ import annotations

from typing import TypeVar, Generic, Optional, Callable

T = TypeVar("T")
R = TypeVar("R")


class OptionException(Exception):
    """
    Custom exception class for signaling errors related to option parsing or handling.
    This exception is a subclass of the built-in Exception class and is designed to be raised
    specifically when an error occurs in the context of option parsing, configuration,
    or argument handling in a program or library.
    
    Attributes:
        Inherits all attributes from the base Exception class without any additions.
    
    Methods:
        Inherits all methods from the base Exception class without any modifications.

    """
    pass


class Option(Generic[T]):
    """
    A Generic Option class that encapsulates an optional value. It is a type-safe way of representing optional (i.e., nullable) objects of a given type T. An instance of Option can contain a value (represented as '_some') of the specified type or can be empty (None).
    This design mimics some of the safety features found in languages like Rust, where the presence of a value must be explicitly handled, thus avoiding null pointer exceptions. An Option must either contain a value (some) or contain nothing (none), and provides methods to handle these cases.
    
    Attributes:
        _some (Optional[T]):
             The encapsulated value that can be of any type T, or None.
    
    Methods:
        __init__(self, some:
             Optional[T]): Initializes the Option with a value of type T or None.
        unwrap(self, message:
             Optional[str]=None) -> T: Returns the contained value if it exists, otherwise raises an OptionException with a provided message.
        unwrap_or(self, default:
             T) -> T: Returns the contained value if it exists, otherwise returns a provided default value.
        map_or(self, map_func:
             Callable[[T], R], default: R) -> R: Applies a function to the contained value if it exists, otherwise returns a provided default value.
        as_optional(self) -> Optional[T]:
             Returns the possibly-contained value as a standard Python optional (i.e., a value or None).
        is_none(self) -> bool:
             Returns True if the Option contains no value.
        is_some(self) -> bool:
             Returns True if the Option contains a value.
        some(some:
             T) -> Option[T]: A static method that creates and returns an Option containing a given value.
        none() -> Option[T]:
             A static method that creates and returns an empty Option.
        __repr__(self) -> str:
             Returns the machine-readable string representation of the Option instance.
        __str__(self) -> str:
             Returns the human-readable string representation of the Option instance.

    """

    _some: Optional[T]

    def __init__(self, some: Optional[T]):
        """
        Initializes an instance of the class.
        This constructor method sets the initial state of the object by assigning the provided
        argument to an instance variable.
        
        Args:
            some (Optional[T]):
                 An optional parameter of generic type T, which is to be assigned
                to the instance variable `_some`. If `None`, `_some` is not set.

        """
        self._some = some

    def unwrap(self, message: Optional[str] = None) -> T:
        """
        Unwraps the value contained within a class instance if it is not None, otherwise raises an `OptionException` with an optional message.
        
        Args:
            message (Optional[str], optional):
                 A custom message to provide when raising an `OptionException`. Defaults to None, in which case an empty string is used as the message.
        
        Returns:
            (T):
                 The value contained within the class instance if it is not None.
        
        Raises:
            OptionException:
                 If the contained value is None, this exception is raised with the provided message or an empty string if no message was provided.

        """
        if self._some is not None:
            return self._some

        raise OptionException(message if message is not None else "")

    def unwrap_or(self, default: T) -> T:
        """
        
        Returns the contained value or a default if the contained value is None.

        """
        if self._some is not None:
            return self._some
        return default

    def map_or(self, map_func: Callable[[T], R], default: R) -> R:
        """
        Applies a function to the contained value if it exists, otherwise returns a default value.
        This method takes a function `map_func` that will be applied to the contained value
        if it is not `None`. If the contained value is `None`, this method returns the provided
        `default` value without applying the function.
        
        Args:
            map_func (Callable[[T], R]):
                 The function to apply to the contained value. The
                function should accept a single argument of type `T` and return a value of type `R`.
            default (R):
                 The default value to return if the contained value is `None`.
        
        Returns:
            (R):
                 The result of applying `map_func` to the contained value if it exists;
                otherwise, returns the `default` value.
            

        """

        if self._some is not None:
            return map_func(self._some)
        return default

    def as_optional(self) -> Optional[T]:
        """
        
        Returns the optional value of the instance's '_some' attribute.
            This method allows the caller to retrieve the value contained within the '_some' attribute,
            wrapped in an Optional type. An Optional type is a type hint that specifies that the return value
            might be a certain type (T) or None. This is useful for functions that might not always have
            a meaningful value to return and hence could return None when there's nothing to return.
        
        Returns:
            (Optional[T]):
                 The value of the '_some' attribute, which can be of any type (T),
                or None if the attribute has no value (equivalent to the absence of a value).

        """
        return self._some

    def is_none(self) -> bool:
        """
        Checks if the internal variable `_some` is `None`.
        
        Returns:
            (bool):
                 True if `_some` is None, False otherwise.

        """
        return self._some is None

    def is_some(self) -> bool:
        """
        
        Returns a boolean indicating whether the object is not None.
            This method inversely checks if the internal state of an object is not 'None'. If the object has any
            value other than 'None', it will return True, otherwise False.
        
        Returns:
            (bool):
                 True if the object is not None, False otherwise.
            

        """
        return not self.is_none()

    @staticmethod
    def some(some: T) -> Option[T]:
        """
        Generates an instance of `Option` containing a value.
        This static method creates an `Option` object encapsulating the given value.
        The result is an `Option` instance that contains a value (is `Some`), as
        opposed to a `None` value which would indicate absence of a value.
        
        Args:
            some (T):
                 The value to be encapsulated by the `Option` object.
        
        Returns:
            (Option[T]):
                 An `Option` instance containing the provided value.

        """
        return Option(some)

    @staticmethod
    def none() -> Option[T]:
        """
        Generates an Option instance representing 'none', which does not contain a value.
        
        Returns:
            (Option[T]):
                 An Option object instantiated with 'None', representing the absence of a value.

        """
        return Option(None)

    def __repr__(self) -> str:
        """
        
        Attributes:
            is_none (Callable):
                 Function to check if the option is None.
            _some (Any):
                 Internal value of the option if it's not None.
        
        Returns:
            (str):
                 A string representation of the option.
            (Description):
                Generate a string representation of the option instance.
                This method provides a user-friendly string that describes the state of the option object.
                It returns 'Option(None)' if the option is None, otherwise it displays 'Option(some_value)',
                where some_value is the internal value held by the option.

        """
        return "Option(None)" if self.is_none() else f"Option({self._some})"

    def __str__(self) -> str:
        """
        
        Returns a human-readable string representation of the object.
            If the object's `is_none` method returns True, it returns the string 'none'.
            Otherwise, it returns a formatted string that contains the value of the object's `_some` attribute.
        
        Returns:
            (str):
                 The string representation of the object.
            

        """
        return "none" if self.is_none() else f"{self._some}"
