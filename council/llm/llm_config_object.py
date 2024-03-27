"""

A module for representing and manipulating configuration objects for Language Learning Models (LLM).

This module contains a set of classes for defining and working with configuration specifications for different LLM providers such as OpenAI, Azure, and Anthropic.

Classes:
    LLMProviders(Enum): An enumeration that defines the available LLM providers.

    LLMProvider: Encapsulates the details and specifications of a language learning model provider.

    LLMConfigSpec(DataObjectSpecBase): Represents the specifications for a LLM configuration including provider, fallback provider, and other parameters.

    LLMConfigObject(DataObject[LLMConfigSpec]): Represents a LLM configuration object that can be initialized from a dictionary or loaded from a YAML file.

The module also includes related utility functions and exception handling for missing or invalid configuration data.


"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Optional

import yaml

from council.utils import DataObject, DataObjectSpecBase
from council.utils.parameter import Undefined


class LLMProviders(str, Enum):
    """
    An enumeration of Large Language Model (LLM) providers.
    This class is a subclass of the `str` and `Enum` classes, and it defines
    enumeration members that represent different Large Language Model providers.
    Each enumeration member has a name that identifies the LLM provider and a
    string value that represents its specific specification identifier.
    
    Attributes:
        OpenAI (str):
             An enumeration member representing the OpenAI specifications.
        Azure (str):
             An enumeration member representing the Microsoft Azure specifications.
        Anthropic (str):
             An enumeration member representing the Anthropic specifications.

    """
    OpenAI = "openAISpec"
    Azure = "azureSpec"
    Anthropic = "anthropicSpec"


class LLMProvider:
    """
    A provider class for Large Language Models (LLMs) that encapsulates the provider's details, specifications, and provider type.
    
    Attributes:
        name (str):
             The name of the LLM provider.
        description (str):
             A brief description of the LLM provider.
        _specs (Dict[str, Any]):
             The specifications and configurations of the LLM provider.
        _kind (LLMProviders):
             The kind of the LLM provider, an enumeration of supported providers.
        Properties:
        kind:
             Returns the kind of LLM provider as an LLMProviders enum.
    
    Methods:
        is_of_kind(kind:
             LLMProviders) -> bool:
            Checks whether the provider is of a specific kind.
        from_dict(values:
             Dict[str, Any]) -> 'LLMProvider':
            Class method to create an instance of LLMProvider from a dictionary.
        to_dict() -> Dict[str, Any]:
            Serializes the LLMProvider instance into a dictionary.
        must_get_value(key:
             str) -> Any:
            Retrieves a value from the specifications dictionary forcefully, failing if not found.
        get_value(key:
             str, required: bool=False, default: Optional[Any]=Undefined()) -> Optional[Any]:
            Retrieves a value from the specifications dictionary with the ability to specify if it is required or provide a default value.
        __str__() -> str:
            Provides a human-readable string representation of the LLMProvider instance.

    """
    def __init__(self, name: str, description: str, specs: Dict[str, Any], kind: LLMProviders):
        """
        Initialize a new object instance with provided name, description, specifications, and kind.
        
        Args:
            name (str):
                 A unique name given to the object.
            description (str):
                 A brief summary or explanation of the object's purpose.
            specs (Dict[str, Any]):
                 A dictionary containing specifications and details about the object.
            kind (LLMProviders):
                 An enumeration member representing the kind of language model provider.
        
        Attributes:
            name (str):
                 Human-readable name identifying the object.
            description (str):
                 Additional information or context for the object's usage or behavior.
            _specs (Dict[str, Any]):
                 Internal storage of specifications and attributes of the object.
            _kind (LLMProviders):
                 Internal marker indicating the type of language model provider.
            

        """
        self.name = name
        self.description = description
        self._specs = specs
        self._kind = kind

    @property
    def kind(self) -> LLMProviders:
        """
        Gets the provider kind of the current instance.
        
        Returns:
            (LLMProviders):
                 An enum representing the specific kind of language model provider.

        """
        return self._kind

    def is_of_kind(self, kind: LLMProviders) -> bool:
        """
        Check if the object's kind matches the provided kind.
        
        Args:
            kind (LLMProviders):
                 An enum member representing a specific provider.
        
        Returns:
            (bool):
                 True if the object's kind is the same as the provided kind, False otherwise.
            

        """
        return self._kind == kind

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMProvider:
        """
        Creates a new LLMProvider instance from the given dictionary of values.
        This class method initializes an LLMProvider object by extracting the name,
        description, and specifications from a dictionary corresponding to one of the
        known LLMProviders (i.e., OpenAI, Azure, Anthropics). It raises a ValueError if
        the dictionary does not contain specifications for a supported LLMProvider.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary containing keys such as 'name',
                'description', and a key corresponding to a supported LLMProvider, which
                itself is expected to be a dictionary of specifications.
        
        Returns:
            (LLMProvider):
                 An instance of LLMProvider configured with the extracted
                data.
        
        Raises:
            ValueError:
                 If the dictionary does not include specifications for any
                of the supported LLMProviders.

        """
        name = values.get("name", "")
        description = values.get("description", "")

        spec = values.get(LLMProviders.OpenAI)
        if spec is not None:
            return LLMProvider(name, description, spec, LLMProviders.OpenAI)
        spec = values.get(LLMProviders.Azure)
        if spec is not None:
            return LLMProvider(name, description, spec, LLMProviders.Azure)
        spec = values.get(LLMProviders.Anthropic)
        if spec is not None:
            return LLMProvider(name, description, spec, LLMProviders.Anthropic)
        raise ValueError("Unsupported model provider")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the current object's state to a dictionary representation.
        This method serializes the object's attributes into a dictionary format, including
        its name and description. If the object is of a specific kind relating
        to the various LLMProviders, such as OpenAI, Azure, or Anthropic, their respective
        settings (_specs) will be included in the resulting dictionary under their corresponding key.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary with the relevant key-value pairs representing the
                object. Keys will typically include 'name', 'description', and optionally
                one of the LLMProviders' keys with their associated _specs.
            

        """
        result: Dict[str, Any] = {"name": self.name, "description": self.description}
        if self.is_of_kind(LLMProviders.OpenAI):
            result[LLMProviders.OpenAI] = self._specs
        if self.is_of_kind(LLMProviders.Azure):
            result[LLMProviders.Azure] = self._specs
        if self.is_of_kind(LLMProviders.Anthropic):
            result[LLMProviders.Anthropic] = self._specs
        return result

    def must_get_value(self, key: str) -> Any:
        """
        Retrieves the value associated with the specified key from an internal collection. This method ensures that the key exists and will raise an error if the key is not found in the collection. This is a strict method that is used when the value is mandatory and its absence would be considered exceptional. It is a wrapper around the `get_value` method with the `required` parameter explicitly set to `True`.
        
        Args:
            key (str):
                 The key for which to retrieve the value.
        
        Returns:
            (Any):
                 The value associated with the given key.
        
        Raises:
            KeyError:
                 If the key does not exist in the collection.

        """
        return self.get_value(key=key, required=True)

    def get_value(self, key: str, required: bool = False, default: Optional[Any] = Undefined()) -> Optional[Any]:
        """
        Retrieves the value associated with a specified key from the provider's specifications.
        This method tries to obtain a value associated with the given `key` from the provider's `_specs` dictionary. If the value is a dictionary, it may provide additional metadata such as a default value or an associated environment variable. If an environment variable is specified and exists, it will use its value, or, if not set, it will fallback to the specified default.
        
        Args:
            key (str):
                 The key for which to retrieve the value.
            required (bool, optional):
                 Whether the key is required; if True and the key or value is not present, an exception is raised.
            default (Optional[Any], optional):
                 A default value to return if the key is not found. The special sentinel value `Undefined()` is used to detect whether a default was provided by the caller.
        
        Returns:
            (Optional[Any]):
                 The value associated with the specified key, or `default` if the key is not found and `default` is not `Undefined()`.
        
        Raises:
            Exception:
                 If the key is marked as required and is missing from the `_specs`, or if the necessary environment variable is not set and no default is provided.

        """
        maybe_value = self._specs.get(key, None)
        if maybe_value is None:
            if not isinstance(default, Undefined):
                return default

        if isinstance(maybe_value, dict):
            default_value: Optional[str] = maybe_value.get("default", None)
            env_var_name: Optional[str] = maybe_value.get("fromEnvVar", None)
            if env_var_name is not None:
                maybe_value = os.environ.get(env_var_name, default_value)

        if maybe_value is None and required:
            raise Exception(f"LLMProvider {self.name} - A required key {key} is missing.")
        return maybe_value

    def __str__(self):
        """
        
        Returns a formatted string representation of the object.
            This special method is used to provide an informal string representation of the object, which can be useful for
            debugging and logging purposes. It is often used for print statements. The string returned includes the object's
            kind, name, and description in a formatted string.
        
        Returns:
            (str):
                 A string that represents the object, containing the object's kind, name, and description.
            

        """
        return f"{self._kind}: {self.name} ({self.description})"


class LLMConfigSpec(DataObjectSpecBase):
    """
    Class representing a configuration specification for a Language Learning Model (LLM).
    
    Attributes:
        description (str):
             A description of the configuration.
        provider (LLMProvider):
             The primary provider of the LLM.
        fallback_provider (Optional[LLMProvider]):
             An optional fallback provider for the LLM.
        parameters (Dict[str, Any]):
             A dictionary of parameters for the LLM configuration.
    
    Methods:
        __init__:
             Initializes a new instance of the LLMConfigSpec class.
        from_dict:
             Class method to create a LLMConfigSpec instance from a dictionary.
        to_dict:
             Converts the LLMConfigSpec instance to a dictionary.
        __str__:
             Returns a string representation of the LLMConfigSpec instance.
    
    Raises:
        ValueError:
             If the provider is not defined in `from_dict` method.

    """
    def __init__(
        self, description: str, provider: LLMProvider, fallback: Optional[LLMProvider], parameters: Dict[str, Any]
    ) -> None:
        """
        Initializes a new instance of a class that handles the configuration of a language model provider with optional fallback mechanisms.
        
        Args:
            description (str):
                 A brief description of the purpose or functionality of the language model.
            provider (LLMProvider):
                 The primary language model provider to be used.
            fallback (Optional[LLMProvider]):
                 An alternative language model provider to use as a fallback if the primary provider fails.
            parameters (Dict[str, Any]):
                 Configuration parameters as key-value pairs that define the behavior of the language model provider.
        
        Raises:
            ValueError:
                 If any of the provided parameters are invalid or incompatible.

        """
        self.description = description
        self.provider = provider
        self.parameters = parameters
        self.fallback_provider = fallback

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMConfigSpec:
        """
        Creates an instance of LLMConfigSpec from a dictionary of values.
        This class method constructs a new LLMConfigSpec object using information provided in a dictionary. It is intended to offer a way to easily deserialize configuration data into a LLMConfigSpec instance. The dictionary must contain a 'provider' key with sub-dictionary that corresponds to the LLMProvider. Optionally, it can include 'description' and 'parameters', which provide additional context and configuration options for the LLMConfigSpec. A 'fallbackProvider' can also be specified as an optional configuration in case the primary provider fails.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary containing the necessary information to construct a LLMConfigSpec object. This includes 'description' (str), 'parameters' (Dict[str, Any]), and 'provider' (Dict[str, Any]) as mandatory items, and optionally a 'fallbackProvider' (Dict[str, Any]).
        
        Returns:
            (LLMConfigSpec):
                 An instance of LLMConfigSpec initialized with the data provided in the input dictionary.
        
        Raises:
            ValueError:
                 If the 'provider' key is not present in the values dictionary or if the construction of the LLMProvider from the provided dictionary fails.

        """
        description = values.get("description", "")
        parameters = values.get("parameters", {})
        fallback_spec: Optional[Dict[str, Any]] = values.get("fallbackProvider", None)
        fallback = LLMProvider.from_dict(fallback_spec) if fallback_spec is not None else None
        provider = LLMProvider.from_dict(values["provider"])
        if provider is None:
            raise ValueError("provider needs to be defined.")

        return LLMConfigSpec(description, provider, fallback, parameters)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the current instance to a dictionary representation.
        This method constructs a dictionary that represents the data of the object. It includes key-value pairs for 'description', 'provider',
        and 'parameters'. If the object has a 'fallback_provider', it is also included in the output dictionary.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing the object's properties with appropriate values.

        """
        result = {"description": self.description, "provider": self.provider, "parameters": self.parameters}
        if self.fallback_provider is not None:
            result["fallback_provider"] = self.fallback_provider
        return result

    def __str__(self):
        """
        
        Returns a string representation of the instance with its description attribute.
            This magic method is used to get a user-friendly string representation of an instance which includes the content of the description attribute of the class.
        
        Returns:
            (str):
                 A string that contains the value of the description attribute.

        """
        return f"{self.description}"


class LLMConfigObject(DataObject[LLMConfigSpec]):
    """
    A configuration object class for LLM (Large Language Model) that handles creating instances from both dictionaries and YAML files. It inherits from a generic `DataObject` that is specialized with `LLMConfigSpec` type, ensuring the configuration adheres to a specific structure and type as required by LLM configurations.
    
    Attributes:
        Inherits attributes from the `DataObject` class specialized with `LLMConfigSpec`.
    
    Methods:
        from_dict(cls, values:
             Dict[str, Any]) -> LLMConfigObject:
            Class method that constructs an `LLMConfigObject` instance from a dictionary. It sanity checks the dictionary using `LLMConfigSpec` to ensure correct structure and data types before instantiation.
        from_yaml(cls, filename:
             str) -> LLMConfigObject:
            Class method for creating an `LLMConfigObject` instance from a YAML file. It reads the file, loads the YAML content, and performs a kind check to ensure the configuration is appropriate for LLM before delegating to the `from_dict` method to construct the object.
            It is assumed that the `DataObject` base class already provides mechanism such as `_from_dict` to handle the initialization from a dictionary and `_check_kind` to validate the kind specified within the dictionary or YAML content against an expected kind value.

    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMConfigObject:
        """
        
        Returns an instance of LLMConfigObject created from a dictionary of values.
            This class method creates and returns a new instance of `LLMConfigObject` using the provided dictionary. It utilizes the superclass's `_from_dict` method, which should correctly interpret and map the provided values to the appropriate fields defined in `LLMConfigSpec`. It is expected that the keys in the `values` dictionary match the attribute names defined within `LLMConfigSpec`.
        
        Args:
            values (Dict[str, Any]):
                 A dictionary containing key-value pairs of configuration attributes to be used to instantiate `LLMConfigObject`.
        
        Returns:
            (LLMConfigObject):
                 A new instance of `LLMConfigObject` initialized with values from the input dictionary.
        
        Raises:
            Any exception that can be raised by the superclass `_from_dict` method if the input is invalid or incomplete.
            

        """
        return super()._from_dict(LLMConfigSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMConfigObject:
        """
        Parse a YAML file to create a LLMConfigObject.
        This class method opens a YAML file, parses its contents and validates that it represents the expected kind
        by calling a class-private method to check its kind attribute. If the contents are valid, it constructs
        a LLMConfigObject using the parsed values.
        
        Args:
            filename (str):
                 The path to the YAML configuration file to be parsed.
        
        Returns:
            (LLMConfigObject):
                 An instance of LLMConfigObject populated with configuration values from the parsed YAML file.
        
        Raises:
            FileNotFoundError:
                 If the YAML file specified does not exist.
            yaml.YAMLError:
                 If there is an error parsing the YAML file.
            ValueError:
                 If the contents of the file do not represent an 'LLMConfig' kind.

        """
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMConfig")
            return LLMConfigObject.from_dict(values)
