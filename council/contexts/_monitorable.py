"""

Module _monitorable

This module is responsible for providing a framework to create monitorable
objects that can be rendered into different representations such as text and
JSON. It includes the Monitorable class which acts as the base class for
objects that are intended to be monitored.

Classes:
    Monitorable: A class that represents a monitorable entity with monitor
    management capabilities.

This module also establishes a generic type variable T for Monitorable subclasses
and T_monitored for Monitored subclasses, enabling the use of generic type
annotations.

Functions:
    This module does not contain standalone functions.

Typical usage example:

    class MyMonitorable(Monitorable):
        ...

    monitorable_instance = MyMonitorable('BaseType')
    monitorable_instance.render_as_json()


"""
import json
from typing import Any, Dict, Iterable, List, TypeVar

from ._monitor import Monitor
from ._monitored import Monitored

T = TypeVar("T", bound="Monitorable")
T_monitored = TypeVar("T_monitored", bound="Monitored")


class Monitorable:
    """
    A class that represents an object which is monitorable, containing methods to create and manage monitors.
    The `Monitorable` class wraps an instance of `Monitor` and provides a high level interface to create new monitors for items of generic type `T`, register pre-existing monitored items, and render monitoring information in various formats such as text, dictionary, and JSON.
    
    Attributes:
        _monitor (Monitor):
             A monitor associated with the Monitorable instance.
    
    Methods:
        __init__:
             Constructor initializing with a base type for monitoring.
        monitor:
             Property that returns the associated `Monitor` instance.
        new_monitor:
             Creates a `Monitored` object for a given item with a specified name.
        register_monitor:
             Registers an existing `Monitored` object with the monitor.
        new_monitors:
             Creates a list of `Monitored` objects for an iterable of items with a base name.
        _register_child:
             Helper method to register a child monitor.
        render_as_text:
             Renders the monitor's information as plain text.
        render_as_dict:
             Renders the monitor's information as a dictionary.
        render_as_json:
             Renders the monitor's information in JSON format.
            Note that T and T_monitored denote type variables and are used placehold generic type references.

    """
    def __init__(self, base_type: str):
        """
        Initializes a new instance of the Monitor with the specified `inner` object and `base_type`.
        The `__init__` method sets up the internal state of the Monitor object by assigning the class name of the
        `inner` object to `_type`, setting up `_children` as an empty dictionary, initializing `_properties` as
        an empty dictionary, and storing the provided `base_type`.
        
        Args:
            inner (object):
                 The object that the Monitor instance will encapsulate. Its class name will be used as the Monitor's `_type`.
            base_type (str):
                 A string representing the base type of the Monitor, used for identification or classification.
            

        """
        self._monitor = Monitor(self, base_type)

    @property
    def monitor(self) -> Monitor:
        """
        Property that gets the current Monitor instance.
        This property method will return the monitor instance that is
        currently associated with the object. It is meant to provide
        read-only access to the internal monitor state, allowing for operations
        that require the state of the monitor without modifying it.
        
        Returns:
            (Monitor):
                 The Monitor instance currently associated with this object.
            

        """
        return self._monitor

    def new_monitor(self, name: str, item: T) -> Monitored[T]:
        """
        Creates and returns a new `Monitored` object for a given item.
        This function takes a name and an item of any type, registers the item as a child with the given name,
        and then returns a new Monitored object that encapsulates the item.
        
        Args:
            name (str):
                 The name associated with the item to be monitored.
            item (T):
                 The item to be monitored. The type of the item is indicated by the generic type parameter `T`.
        
        Returns:
            (Monitored[T]):
                 A `Monitored` instance encapsulating the given item.
        
        Raises:
            ValueError:
                 If the name provided is not valid or if the item cannot be registered.
                Note that the process of registering a child might vary and could include checks or other logic
                to ensure that the item is suitable for being monitored. This might raise exceptions based on those checks.
                This docstring assumes that `_register_child` is a method meant to handle registration logic,
                including potential validation, and that `Monitored` is a generic class designed to monitor a given item.

        """
        self._register_child(name, item)
        return Monitored(name, item)

    def register_monitor(self, monitored: T_monitored) -> T_monitored:
        """
        Registers a monitor instance to the system.
        This method takes a monitor object, registers it with the system by adding it to an internal
        registry, and then returns the same monitor object. The method relies on two attributes of the
        monitored object: its 'name' and 'inner' attribute. The 'name' is used as the key while the 'inner'
        attribute is the value for the internal registry.
        
        Args:
            monitored (T_monitored):
                 The monitor instance that needs to be registered.
                'T_monitored' should be a type with at least 'name' and 'inner' attributes.
        
        Returns:
            (T_monitored):
                 The same monitor instance that was passed in after it has been registered.

        """
        self._register_child(monitored.name, monitored.inner)
        return monitored

    def new_monitors(self, name: str, items: Iterable[T]) -> List[Monitored[T]]:
        """
        Creates a list of Monitored instances from an iterable of items.
        Given a name and an iterable, this method will instantiate a Monitored object for each element in the iterable,
        given them unique names by appending their index in the iterable to the provided base name, and then registers
        the newly created Monitored instances as children to the current monitoring structure.
        
        Args:
            name (str):
                 The base name to be used for each Monitored instance.
            items (Iterable[T]):
                 An iterable of items to be monitored.
        
        Returns:
            (List[Monitored[T]]):
                 A list of Monitored instances, one for each item in the provided iterable.

        """
        result = [Monitored(f"{name}[{index}]", item) for index, item in enumerate(items)]
        [self._register_child(item.name, item.inner) for item in result]
        return result

    def _register_child(self, relation: str, child: "Monitorable"):
        """
        Registers a child object to the monitorable object's monitoring structure.
        This method ties a child `Monitorable` instance to the current parent `Monitorable` instance within the monitoring hierarchy through the specified relation.
        It adds the child's `_monitor` attribute to the parent's `_monitor` relationships. This is typically an internal method used to maintain the structure of the monitoring graph.
        
        Args:
            relation (str):
                 The type of relationship to establish between the parent and child within the monitoring context.
            child ('Monitorable'):
                 The child monitorable object that will be registered under the parent for monitoring purposes.
            

        """
        self._monitor.register_child(relation, child._monitor)

    def render_as_text(self) -> str:
        """
        Renders the current state of the monitor as a text string with each component separated by new lines.
        
        Returns:
            (str):
                 A string representation of the monitor's current state with each component on a new line.

        """
        return "\n".join(self.monitor.render_as_text())

    def render_as_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """
        Renders the object as a dictionary representation, optionally including its children's renderings.
        
        Args:
            include_children (bool, optional):
                 A flag indicating whether to include the rendering of child elements in the dictionary representation. Defaults to True.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing the rendered state of the object, and if specified, its children.

        """
        return self.monitor.render_as_dict(include_children)

    def render_as_json(self) -> str:
        """
        Renders the monitor's current state as a JSON-formatted string.
        This method serializes the monitor's current state into a JSON string
        by converting it to a dictionary and then using `json.dumps` with an
        indentation of 2 spaces for better readability.
        
        Returns:
            (str):
                 A JSON-formatted string representing the state of the monitor.
            

        """
        return json.dumps(self.monitor.render_as_dict(), indent=2)
