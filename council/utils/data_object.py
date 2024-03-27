"""

Module provides a framework for defining and handling data object specifications and metadata.

Classes:
    DataObjectMetadata: Encapsulates common metadata properties for data objects.
    DataObjectSpecBase: Abstract base class defining the interface for specifying data objects.
    DataObject: Generic data object class that combines metadata and specifications.

This module allows users to create, process, and serialize data objects that
contain both metadata and a strongly typed specification. It includes functionality
to convert to and from dictionary representations, facilitating interaction with
YAML configurations.

A typical use case involves defining concrete implementations of DataObjectSpecBase
for specific types of data objects, and instantiating the generic DataObject class
with these specialized specifications.

Functions within this module are responsible for ensuring the consistency and
integrity of data objects, such as matching expected 'kind' types and extracting or
applying labels from metadata.


"""
from __future__ import annotations
from typing_extensions import Self

import abc
from typing import Any, Dict, TypeVar, Generic, Type, Optional

import yaml


class DataObjectMetadata:
    """
    A class that encapsulates metadata for a data object, including its name, description, and a directory of labels.
    This class provides methods to interact with the metadata, such as checking the existence of labels, retrieving label values,
    matching provided labels, and representing the metadata as a dictionary.
    
    Attributes:
        name (str):
             The name of the data object.
        description (Optional[str]):
             A description of the data object. Default is None if not provided.
        labels (Dict[str, Any]):
             A dictionary of labels associated with the data object.
    
    Methods:
        has_label(label:
             str) -> bool:
            Checks if the specified label exists within the labels.
        get_label_value(label:
             str) -> Optional[Any]:
            Retrieves the value for the specified label if it exists, else returns None.
        is_matching_labels(labels:
             Dict[str, Any]) -> bool:
            Determines if the provided dictionary of labels matches the object's labels.
        to_dict() -> Dict[str, Any]:
            Converts the metadata into a dictionary format.
        from_dict(values:
             Dict[str, Any]) -> DataObjectMetadata:
            Creates an instance of `DataObjectMetadata` from a dictionary.
            The `from_dict` is a class method that takes a dictionary and returns a new instance of `DataObjectMetadata`
            based on the provided values. It looks for 'name', 'labels', and 'description' keys in the dictionary
            to initialize the respective attributes of the class. The 'labels' and 'description' keys are optional.

    """
    def __init__(self, name: str, labels: Dict[str, Any], description: Optional[str] = None):
        """
        Initializes a new instance with a name, labels, and an optional description.
        
        Args:
            name (str):
                 The name identifier for the instance.
            labels (Dict[str, Any]):
                 A dictionary of labels containing metadata about the instance.
            description (Optional[str], optional):
                 A text description for the instance. Defaults to None.
        
        Attributes:
            name (str):
                 The name identifier for the instance, as provided by the 'name' argument.
            description (Optional[str]):
                 The description for the instance, which can be `None` if not provided.
            labels (Dict[str, Any]):
                 A dictionary of labels with additional information about the instance, as provided by the 'labels' argument.
            

        """
        self.name = name
        self.description = description
        self.labels = labels

    def has_label(self, label: str) -> bool:
        """
        Determines whether a specified label exists within the object's labels attribute.
        
        Args:
            label (str):
                 The label to check for existence within the labels.
        
        Returns:
            (bool):
                 True if the label is present in the labels, False otherwise.

        """
        return label in self.labels

    def get_label_value(self, label: str) -> Optional[Any]:
        """
        Retrieves the value associated with a given label, if it exists.
        This method looks for the provided `label` in the label dictionary of the current object. If the label is found, the associated value is returned. Otherwise, the method returns None, indicating that the label does not exist within the labels dictionary.
        
        Args:
            label (str):
                 The label for which the value needs to be fetched.
        
        Returns:
            (Optional[Any]):
                 The value associated with the provided label, or None if the label is not in the dictionary.
            

        """
        if label in self.labels:
            return self.labels[label]
        return None

    def is_matching_labels(self, labels: Dict[str, Any]) -> bool:
        """
        Checks if the provided labels match the object's labels.
        This method iterates over a dictionary of labels, comparing each with the value
        obtained from the object for that label. If all the provided labels match the
        corresponding values in the object, the method returns `True`. Otherwise, it
        returns `False` if there is at least one mismatch.
        
        Args:
            labels (Dict[str, Any]):
                 A dictionary where keys are label names and values are
                the expected label values to match against the object's label values.
        
        Returns:
            (bool):
                 `True` if all provided labels match the object's label values, otherwise `False`.
            

        """
        for label in labels:
            value = self.get_label_value(label)
            if value != labels[label]:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation including essential information.
        This method collects the object's attributes and assembles them into a dictionary with keys representing
        attribute names and their corresponding values. It will always include the 'name' and 'labels' attributes.
        If the 'description' attribute is present and not None, it will be included in the resulting dictionary.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representation of the object with keys for 'name', 'labels', and
                optionally 'description' if it is not None.

        """
        result = {"name": self.name, "labels": self.labels}

        if self.description is not None:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> DataObjectMetadata:
        """
        Class method that creates a new instance of `DataObjectMetadata` from a dictionary.
        This method constructs a new `DataObjectMetadata` object using the provided dictionary. It expects the dictionary to have a 'name' key and optionally 'labels' and 'description' keys. If 'labels' or 'description' are not provided, they will default to an empty dictionary and `None`, respectively.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary containing keys with information to construct a `DataObjectMetadata` object. Mandatory key is 'name'. 'labels' and 'description' are optional.
        
        Returns:
            (DataObjectMetadata):
                 A new `DataObjectMetadata` instance initialized with the values from the input dictionary.

        """
        return DataObjectMetadata(values["name"], values.get("labels", {}), values.get("description", None))


T = TypeVar("T", bound="DataObjectSpecBase")


class DataObjectSpecBase(abc.ABC):
    """
    A base abstract class that defines a specification for data objects.
    This abstract class provides a contract for subclasses to implement serialization and deserialization
    methods for converting between dictionary representations and instances of the subclass.
    The class methods `to_dict` and `from_dict` are defined as abstract methods, enforcing subclasses
    to provide concrete implementations for them.
    
    Attributes:
        None.
    
    Methods:
        to_dict:
             Abstract method that when implemented, should serialize the instance into a dictionary
            with string keys and values of any type.
        from_dict:
             Abstract method that when implemented, should create an instance of the implementing
            class from a dictionary with string keys and values of any type. This class method serves as a factory method.

    """
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation.
        This method should be implemented by subclasses to ensure that each object can
        be represented as a dictionary. The keys of the dictionary should correspond
        to attribute names, and the values should reflect the attribute values.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representation of the object's attributes.
            

        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, values: Dict[str, Any]) -> Self:
        """
        Converts a dictionary of values into an instance of the class.
        This class method takes a dictionary where keys correspond to attributes of the class,
        and values are the data that these attributes should be set to. It returns a new
        instance of the class with the attributes set to the provided values.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary where each key-value pair represents
                the name of the attribute and the value it should be set to.
        
        Returns:
            (Self):
                 An instance of the class populated with the data from the input dictionary.
        
        Raises:
            NotImplementedError:
                 If the method is not implemented in a subclass.
            

        """
        pass


class DataObject(Generic[T]):
    """
    A generic data structure that encapsulates a versioned object with metadata and a typed specification.
    The `DataObject` class is designed to hold data structures conforming to a specific schema, characterized by a `kind` and a `version`. It also includes `metadata` which provides additional information about the instance, and a typed `spec` which contains the actual specifications or settings of the object.
    
    Attributes:
        kind (str):
             A string identifier for the kind of object.
        version (str):
             A version string identifying the version of the object structure.
        metadata (DataObjectMetadata):
             An instance of `DataObjectMetadata` containing metadata about the object.
        spec (T):
             A generic type `T` indicating the object's specifications or detailed configuration.
    
    Methods:
        to_dict(self) -> Dict[str, Any]:
            Serializes the `DataObject` instance into a dictionary.
        _check_kind(cls, values:
             Dict[str, Any], expected: str) -> None:
            Class method that validates whether the kind specified in a dictionary matches an expected kind.
    
    Raises:
        ValueError:
             If the kinds do not match.
        _from_dict(cls, inner:
             Type[T], values: Dict[str, Any]) -> 'DataObject[T]':
            Class method that creates an instance of `DataObject` from a dictionary.
        to_yaml(self, filename:
             str):
            Serializes the `DataObject` instance into a YAML file.
    
    Note:
        The `DataObject` class is parametrized with a type variable `T` which allows for type-safe storage
        of different specification objects. The generic type T must have a `from_dict` method implemented
        to enable deserialization from a dictionary.

    """
    def __init__(self, kind: str, version: str, metadata: DataObjectMetadata, spec: T):
        """
        Initializes a new instance of a resource with specific kind, version, metadata, and specification.
        
        Args:
            kind (str):
                 A string representing the resource type.
            version (str):
                 A string specifying the version of the resource.
            metadata (DataObjectMetadata):
                 An instance of DataObjectMetadata containing the resource's metadata.
            spec (T):
                 The specification of the resource, where T is a bounded type representing the specification.

        """
        self.kind = kind
        self.version = version
        self.metadata = metadata
        self.spec = spec

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object's attributes to a dictionary representation.
        This method traverses the object's attributes, converting them into a dictionary
        that represents the object's state. The 'kind', 'version', and 'metadata' attributes
        are straightforwardly represented while 'metadata' and 'spec' attributes,
        which are presumably objects with their own to_dict methods, are converted
        to dictionaries by calling their respective to_dict methods.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing keys 'kind', 'version', 'metadata', and 'spec'
                with their corresponding values as the object's current state.
            

        """
        return {
            "kind": self.kind,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def _check_kind(cls, values: Dict[str, Any], expected: str) -> None:
        """
        Checks if the provided `kind` within a dictionary matches the `expected` kind.
        This class method ensures that the `kind` field in the given `values` dictionary corresponds to the specified `expected` value. If the `kind` does not match, it raises a ValueError.
        
        Args:
            values (Dict[str, Any]):
                 The dictionary containing the `kind` field to check.
            expected (str):
                 The expected value for the `kind` field.
        
        Raises:
            ValueError:
                 If the `kind` field in `values` does not match the `expected` value.

        """
        kind = values.get("kind", None)
        if kind != expected:
            raise ValueError(f"Expected kind: `{expected}`, found `{kind}` instead.")

    @classmethod
    def _from_dict(cls, inner: Type[T], values: Dict[str, Any]) -> Self:
        """
        Converts a dictionary to an instance of the class.
        This class method takes a dictionary representation of the object and converts it into an instance of the class.
        It extracts the 'metadata' and 'spec' keys from the provided dictionary and uses the associated inner Type provided as an argument to construct the object.
        
        Args:
            inner (Type[T]):
                 The inner Type that is used to construct the 'spec' part of the class.
            values (Dict[str, Any]):
                 The dictionary representation of the object that contains keys for 'kind', 'version',
                'metadata' (which is itself a dictionary that should be convertible to DataObjectMetadata),
                and 'spec' (which is a dictionary expected to correspond to the structure of the 'inner' Type).
        
        Returns:
            (Self):
                 An instance of the class constructed using the provided dictionary.
        
        Raises:
            KeyError:
                 If any of the required keys ('kind', 'version', 'metadata', or 'spec') are missing from the values dictionary.

        """
        metadata = DataObjectMetadata.from_dict(values["metadata"])
        spec = inner.from_dict(values["spec"])
        return cls(values["kind"], values["version"], metadata, spec)

    def to_yaml(self, filename: str):
        """
        Writes the object's dictionary representation to a YAML file.
        
        Args:
            filename (str):
                 The name of the file where the YAML data will be written to.
        
        Raises:
            OSError:
                 An error occurred when attempting to write to the file.
            yaml.YAMLError:
                 An error occurred during the YAML dumping process.

        """
        values = self.to_dict()
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(values, f, default_flow_style=False)
