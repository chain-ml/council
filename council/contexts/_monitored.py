"""


Module _monitored

This module defines the Monitored class, which is a generic wrapper class intended for monitoring objects. 
The Monitored class encapsulates another object, allowing access to the original object while also providing the ability to track and manage additional metadata or state.

Classes:
    Monitored(Generic[T]): A generic class for creating monitored objects.

Typing:
    T: A type variable that represents the type of the object to be monitored.



"""
from typing import Generic, TypeVar

T = TypeVar("T")


class Monitored(Generic[T]):
    """
    A generic class that wraps an object of type T to monitor it.
    This class serves as a wrapper for any monitorable object, allowing users to
    track and handle the object using its `name` and the underlying object itself
    through the `inner` property. The `Monitored` class is designed to be used
    with any type T, making it flexible for various monitoring scenarios.
    
    Attributes:
        _name (str):
             A private attribute that holds the name of the monitorable object.
        _inner (T):
             A private attribute that contains the monitorable object of type T.
    
    Methods:
        inner:
             Property that returns the monitorable object of type T.
        name:
             Property that returns the name of the monitorable object.
        Type Parameters:
        T:
             The type of the object to be monitored.

    """
    def __init__(self, name: str, monitorable: T):
        """
        Initializes an instance of the class.
        This constructor method initializes the class with a name and a monitorable object.
        The `name` parameter is meant to be a human-readable identifier for the instance,
        while the `monitorable` parameter is an object of type `T` that the instance will
        interact with or monitor in some capacity.
        
        Args:
            name (str):
                 A string representing the name or identifier for the instance.
            monitorable (T):
                 An object of generic type T that is to be associated with the
                instance, potentially for monitoring or interacting purposes.
        
        Attributes:
            _name (str):
                 A private string storing the name or identifier.
            _inner (T):
                 A private attribute that holds the monitorable object.

        """
        self._name = name
        self._inner = monitorable

    @property
    def inner(self) -> T:
        """
        
        Returns the encapsulated '_inner' attribute value of the instance.
            This property method is used to access the value of the '_inner' attribute, which is intended to be 'private' in the sense that it's meant for internal use and not for public interface. The @property decorator indicates that this method can be accessed like an attribute, without the need for calling it as a method.
        
        Returns:
            (T):
                 The value of the '_inner' attribute of the instance.
            

        """
        return self._inner

    @property
    def name(self) -> str:
        """
        Property that gets the name of the object.
        This property is used to retrieve the private _name attribute from an instance of the class.
        It is decorated with @property, making it a getter for the _name attribute, allowing for
        encapsulation by not exposing the attribute directly.
        
        Returns:
            (str):
                 The name value of the instance.
            

        """
        return self._name
