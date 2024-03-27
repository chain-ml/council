"""

Module _monitor

This module defines the class Monitor, which is designed for monitoring and managing the hierarchy
of objects in a structured way. It provides mechanisms to register child monitors,
set and retrieve properties, and render the current state as text or a dictionary.

Classes:
    Monitor: A class that encapsulates monitoring information about an object and its relationships.

Attributes:
    _children (Dict[str, 'Monitor']): Map of relationship names to child Monitor instances.
    _base_type (str): The base type of the object being monitored.
    _properties (Dict[str, Any]): A dictionary of property names and values for the monitored object.

Methods:
    __init__(inner: object, base_type: str): Constructs a Monitor instance with the specified base type.
    register_child(relation: str, child: 'Monitor') -> None: Registers a child Monitor under the specified relation.
    set(name: str, value: Any) -> None: Sets a property by name to the given value.
    render_as_text(prefix: str, indent: int, indent_step: int) -> List[str]: Renders the monitor hierarchy as a list of strings.
    render_as_dict(include_children: bool) -> Dict[str, Any]: Renders the monitor hierarchy as a dictionary.

Properties:
    type (str): Returns the type name of the object being monitored.
    name (str): Gets or sets the name property of the monitored object.
    children (Mapping[str, 'Monitor']): Returns a read-only map of registered child monitors.
    properties (Mapping[str, Any]): Returns a read-only dictionary of the monitored object's properties.
    base_type (str): Returns the base type string of the monitored object.


"""
from typing import Any, Dict, List, Mapping


class Monitor:
    """
    A class representing a Monitor, which can track and render objects along with their relationships and properties.
    This object-oriented class provides a structure for monitoring an object, with functionality to
    register child monitors, set and retrieve properties, and render details as either text or dictionaries.
    The class includes several properties that expose various aspects of the monitored object such as its
    type, name, child monitors, and base type.
    
    Attributes:
        _children (Dict[str, 'Monitor']):
             A dictionary mapping relationship identifiers to child Monitor
            instances.
        _base_type (str):
             The base type of the monitor.
        _properties (Dict[str, Any]):
             A dictionary of all properties and associated values of the monitor.
    
    Methods:
        __init__(self, inner:
             object, base_type: str):
            Initializes a new instance of the Monitor class.
        register_child(self, relation:
             str, child: 'Monitor') -> None:
            Registers a child monitor with a specified relationship identifier.
        set(self, name:
             str, value: Any) -> None:
            Sets a property with the given name and value.
        type(self) -> str:
            Returns the type of the monitor.
        name(self) -> str:
            Gets the name property of the monitor.
        name(self, value:
             str) -> None:
            Sets the name property of the monitor.
        children(self) -> Mapping[str, 'Monitor']:
            Returns a read-only view of the child monitors.
        properties(self) -> Mapping[str, Any]:
            Returns a read-only view of the monitor's properties.
        base_type(self) -> str:
            Returns the base type of the monitor.
        render_as_text(self, prefix:
             str='', indent: int=0, indent_step: int=2) -> List[str]:
            Renders the monitor and its children in a human-readable text format.
        render_as_dict(self, include_children:
             bool=True) -> Dict[str, Any]:
            Renders the monitor and, optionally, its children as a dictionary structure.

    """
    _children: Dict[str, "Monitor"]
    _base_type: str
    _properties: Dict[str, Any]

    def __init__(self, inner: object, base_type: str):
        """
        Initializes a new instance of a class with specified attributes for inner object, children,
        properties, and base type.
        
        Args:
            inner (object):
                 An object from which the type name is extracted and stored in `_type`.
            base_type (str):
                 The name of the base type associated with the instance which is stored in '_base_type'.
        
        Attributes:
            _type (str):
                 The name of the class of the `inner` object passed during initialization.
            _children (dict):
                 A dictionary to store child elements related to the object. Initially empty.
            _properties (dict):
                 A dictionary to store properties related to the object. Initially empty.
            _base_type (str):
                 A string representing the base type of the object, provided as an argument during initialization.

        """
        self._type = inner.__class__.__name__
        self._children = {}
        self._properties = {}
        self._base_type = base_type

    def register_child(self, relation: str, child: "Monitor") -> None:
        """
        Registers a child Monitor to the current object with a specified relation.
        
        Args:
            relation (str):
                 A string representing the type of relation this Monitor has with child.
            child ('Monitor'):
                 An instance of Monitor which is to be registered as a child.
        
        Returns:
            None
        
        Raises:
            TypeError:
                 If 'child' is not an instance of Monitor.
            KeyError:
                 If the 'relation' is already associated with another child Monitor.

        """
        self._children[relation] = child

    def set(self, name: str, value: Any) -> None:
        """
        Sets a property 'name' to the given 'value' in the object's properties dictionary.
        
        Args:
            name (str):
                 The property name to set.
            value (Any):
                 The value to assign to the property 'name'.
        
        Returns:
            None

        """
        self._properties[name] = value

    @property
    def type(self) -> str:
        """
        A property that returns the type of the object.
        This read-only property provides access to the type of the object, allowing the user to retrieve the type information
        stored in the '_type' instance variable.
        
        Returns:
            (str):
                 The type of the object as a string.
            

        """
        return self._type

    @property
    def name(self) -> str:
        """
        
        Returns the name property of the object stored under `_properties` attribute.
        
        Returns:
            (str):
                 The value of the 'name' key from the object's `_properties` dictionary.
        
        Raises:
            KeyError:
                 If the 'name' key does not exist within the `_properties` dictionary.

        """
        return self._properties["name"]

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets the 'name' property of the object to a given value. This setter method allows assigning a new value to the name property, which is stored within the _properties dictionary of the object with the key 'name'. It ensures that the name attribute can be updated after the object's instantiation. The method takes a single parameter, which is expected to be a string, and does not return anything. It is decorated with the @name.setter decorator, indicating that this method is a setter for the 'name' property.
        
        Attributes:
            value (str):
                 The new value to be set for the 'name' property of the object.
        
        Raises:
            TypeError:
                 If the provided value is not of type str.

        """
        self._properties["name"] = value

    @property
    def children(self) -> Mapping[str, "Monitor"]:
        """
        Property that gets a mapping of the children monitors associated with this monitor instance.
        Each child monitor is represented as a value in the mapping, with the corresponding key being a string that
        uniquely identifies that child monitor within the context of the current monitor instance.
        
        Returns:
            (Mapping[str, 'Monitor']):
                 A mapping from string identifiers to 'Monitor' instances representing the
                children of the current monitor.
            

        """
        return self._children

    @property
    def properties(self) -> Mapping[str, Any]:
        """
        A property that returns the properties of the object as a mapping.

        """
        return self._properties

    @property
    def base_type(self) -> str:
        """
        
        Returns the base type of the object as a string. This property method retrieves the value of the private attribute '_base_type'. The base type typically represents the fundamental category or class of the object instance. It is intended to be accessed as a read-only attribute, providing information about the nature of the object without modifying it.
        
        Returns:
            (str):
                 A string representing the base type of the object.

        """
        return self._base_type

    def render_as_text(self, prefix: str = "", indent: int = 0, indent_step: int = 2) -> List[str]:
        """
        Renders the current object as a list of text lines with specified formatting parameters.
        This method takes a prefix, base indentation, and step for indentation, then builds text representations
        of the current object along with its children objects. Each property of the object is listed inline, and
        children are displayed in nested formatting based on the provided indentation step.
        
        Args:
            prefix (str):
                 A string to be prefixed to every line of the text representation. Defaults to an empty string.
            indent (int):
                 The initial indentation level. Defaults to 0.
            indent_step (int):
                 The number of spaces to use for each additional indentation level. Defaults to 2.
        
        Returns:
            (List[str]):
                 A list of strings, where each string is a line of the rendered text representation of the object and its properties.
            

        """
        padding = " " * indent
        properties = ", ".join([f"{name}: {value}" for name, value in self.properties.items()])
        current = f"{padding}{prefix}{self.type}({self.base_type}) {{{properties}}}"
        result = [
            item
            for name, child in self.children.items()
            for item in child.render_as_text(name + ": ", indent + indent_step, indent_step)
        ]

        return [current] + result

    def render_as_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """
        Generates a dictionary representation of the current object, including its properties and optionally its children.
        This method constructs a dictionary that holds key details about the object, such as its properties, type, and base type. If the
        include_children flag is set to True, it will also include a list of dictionaries representing the children of the object.
        
        Args:
            include_children (bool):
                 A flag indicating whether to include the children of the object in the output dictionary. Defaults to True.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing the object's properties, type, base type, and optionally, children.
            

        """
        children = []
        if include_children:
            for name, child in self.children.items():
                children.append({"name": name, "value": child.render_as_dict()})
        return {"properties": self.properties, "type": self.type, "baseType": self.base_type, "children": children}
