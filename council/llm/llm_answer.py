"""

Module that provides functionality for parsing schema-based answers from prompts, YAML blocks, and strings.

This module contains classes and decorators to handle properties and validations for objects defined by schema templates. It includes custom properties with ranking, validation classes, and answer handlers that manage the conversion from text-based representations to Python objects.

Classes:
    LLMParsingException - Custom exception class for parsing errors.
    llm_property - Decorator class for properties with ranking based on the source code line.
    llm_class_validator - Decorator class for class-level validation functions.
    LLMProperty - Encapsulates a schema property, including its name, type, description, rank, and parsing capabilities.
    LLMAnswer - Manages the conversion of text-based prompts into schema-based Python objects, including parsing and validation.

The module provides methods to parse individual lines, YAML blocks, YAML lists, and YAML blocks wrapped in markdown code block syntax. It raises custom exceptions to signal parsing errors and uses a sorting mechanism to maintain the order of properties based on their defined rank.


"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Callable

import yaml

from council.utils import CodeParser


class LLMParsingException(Exception):
    """
    A custom exception for errors encountered during parsing of LLM (Language, Logic, and Meaning) data.
    This exception is derived from the Python's built-in Exception class and is meant to be raised whenever a specific error related to LLM parsing occurs. It does not add any additional functionality to the base Exception class but serves as a distinct type of exception that can be caught and handled separately from other exceptions.
    
    Attributes:
        Inherits all attributes from the base Exception class.

    """
    pass


class llm_property(property):
    """
    A custom property class that extends the built-in property class by including the source line rank of the getter function.
    This class acts as a descriptor and should be used as a decorator for creating managed attributes in a class. It extends the property descriptor, adding an extra attribute 'rank' which indicates the line number on which the getter function is defined within the source code.
    
    Attributes:
        rank (int):
             The line number where the getter function is defined in the source code.
    
    Args:
        fget (callable, optional):
             The getter function for the property. Defaults to None.
        fset (callable, optional):
             The setter function for the property. Defaults to None.
        fdel (callable, optional):
             The deleter function for the property. Defaults to None.
        doc (str, optional):
             The docstring for the property. Defaults to None.
        Inherits:
        property:
             The built-in property class from which this class inherits.
    
    Note:
        This class uses the inspect module to determine the rank (line number) of the getter function which requires the source code to be accessible.

    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        """
        Initializes a new instance of the descriptor for managing attribute access.
        This constructor typically wraps methods for getting, setting, and deleting an attribute, along with an optional
        documentation string. Additionally, it captures the source code line number of the getter function for reference.
        
        Args:
            fget (Optional[Callable]):
                 The function for getting an attribute value.
                Default is None.
            fset (Optional[Callable]):
                 The function for setting an attribute value.
                Default is None.
            fdel (Optional[Callable]):
                 The function for deleting an attribute value.
                Default is None.
            doc (Optional[str]):
                 The docstring for the property attribute. If not provided, the docstring of fget is used.
                Default is None.
        
        Returns:
            None
        
        Raises:
            TypeError:
                 If fget, fset, and fdel are not callable or None.
            OSError:
                 If an error occurs while fetching the source lines of the fget function.

        """
        super().__init__(fget, fset, fdel, doc)
        self.rank = inspect.getsourcelines(fget)[1]


class llm_class_validator:
    """
    A class that serves as a validator by wrapping a given callable function.
    This class is designed to take a function as a parameter during initialization, and it stores that function for later use. This can be useful in scenarios where functions need to be validated or processed before they are executed. The validator can apply certain checks or conditions on the callable passed to it, ensuring that only functions meeting specific criteria are considered valid.
    
    Attributes:
        f (Callable):
             The function that is to be validated or processed. The callable
            should be provided when an instance of the class is created.
    
    Args:
        func (Callable):
             The function that will be wrapped by the llm_class_validator
            for possible validation or further processing.
        

    """
    def __init__(self, func: Callable):
        """
        Initializes an instance of the class, storing a callable object.
        
        Args:
            func (Callable):
                 The callable object to store.
            

        """
        self.f = func


class LLMProperty:
    """
    Class to represent and manage a property for a Machine Learning model.
    
    Attributes:
        _name (str):
             Name of the property.
        _type (type):
             The Python type of the property, inferred from the getter annotation or set to 'str' by default.
        _description (str):
             The documentation string associated with the property.
        _rank (int):
             A ranking or ordering value associated with the property.
    
    Methods:
        name:
             Property that returns the name of the LLM property.
        rank:
             Property that returns the rank of the LLM property.
        __str__:
             Returns a string representation of the LLM property, including its name, description, and expected response type.
        can_parse:
             Determines if a given value can be safely cast to the property's type.
        parse:
             Tries to parse and cast a given value to the property's type, with support for a default value in case of failure.
    
    Raises:
        LLMParsingException:
             An exception indicating that the given value could not be parsed into the expected type.

    """
    def __init__(self, name: str, prop: llm_property):
        """
        Initializes a new instance of the class that this method is declared within. It assigns the provided parameters to the instance's attributes, including the name, type, description, and rank information based on the `llm_property` object provided. The type information is derived from the return type annotation of the property getter method. If no annotation is present, the type will default to `str`. The description is directly obtained from the property's documentation string, and the rank is a predefined attribute of the property object itself. The `__init__` method does not return any value as its purpose is purely for instance initialization. This method is an instance method, which means it is called on an instance of a class and not on the class itself.
        
        Args:
            name (str):
                 The name to assign to the instance's `_name` attribute.
            prop (llm_property):
                 An `llm_property` object representing the property whose details are to be assigned to this instance. Its `fget` method provides type annotations and its `__doc__` provides the textual description.
            

        """
        self._name = name
        self._type = prop.fget.__annotations__.get("return", str)
        self._description = prop.__doc__
        self._rank = prop.rank

    @property
    def name(self) -> str:
        """
        Gets the name of the object instance.
        This property retrieves the private _name attribute
        from the object instance which represents the entity's name.
        It is a read-only property and hence can only be used to
        get the name, not to set it.
        
        Returns:
            (str):
                 The name of the object.

        """
        return self._name

    @property
    def rank(self) -> int:
        """
        Gets the rank of an object.
        This property returns the rank value stored within an instance, which
        is assumed to be an integer. It serves as a getter for the '_rank'
        attribute, ensuring that encapsulation is maintained and direct
        access to the underlying data is not permitted.
        
        Returns:
            (int):
                 The rank value of the instance.

        """
        return self._rank

    def __str__(self):
        """
        
        Returns a formatted string representation of the object with its details.
            This method overrides the default `__str__` method and provides a custom string representation of the object, including its name, description, and expected response type.
        
        Returns:
            (str):
                 A formatted string with the object's name, description, and the expected type of response enclosed in braces.

        """
        return f"{self._name}: {{{self._description}, expected response type `{self._type.__name__}`}}"

    def can_parse(self, value: Any) -> bool:
        """
        Determine if the given value can be parsed by the type parser of the object.
        This method attempts to parse the provided value using the object's internal type parser.
        It returns `True` if the parsing is successful, and `False` otherwise. The parsing may fail
        if the provided value is not of a compatible type or if the value cannot be coerced into
        the expected type, in which case either a `TypeError` or `ValueError` will be raised and
        caught within the method.
        
        Args:
            value (Any):
                 The value to be parsed by the object's type parser.
        
        Returns:
            (bool):
                 A boolean indicating whether the value can be successfully parsed.
            

        """
        try:
            _ = self._type(value)
            return True
        except (TypeError, ValueError):
            return False

    def parse(self, value: Any, default: Optional[Any]) -> Any:
        """
        Parses a given value into a specified type or returns a default value if parsing fails.
        
        Args:
            value (Any):
                 The value to be parsed.
            default (Optional[Any]):
                 The default value to use if parsing fails. If this is `None`,
                a `LLMParsingException` will be raised instead of returning a default value.
        
        Returns:
            (Any):
                 The parsed value if the parsing is successful, or the default value if provided
                and parsing is unsuccessful.
        
        Raises:
            LLMParsingException:
                 If the value cannot be parsed into the specified type and no
                default value is provided.
                The function attempts to convert the provided `value` into the data type specified by
                `self._type`. The conversion strategy is type-dependent; for bool types, it employs
                a custom `converter` function that recognizes specific string representations of
                boolean values. For other types, it relies on the direct casting capability of the type.
                If parsing is successful, the parsed value is returned. If the parsing fails due to
                TypeError or ValueError, the function will either return the provided `default` value if
                it is not `None`, or it will raise a `LLMParsingException` detailing the failure if
                no default value is provided.
                This method is typically meant to be used within classes that handle the parsing and
                conversion of configuration values or similar settings.

        """
        def converter(x: str) -> bool:
            """
            Converts a string input to a boolean value based on specific string equivalents.
            
            Args:
                x (str):
                     The string input to convert to a boolean. The string is expected to
                    represent a truthful value ('true', '1', 't') or a false value
                    ('false', '0', 'f') after being stripped of leading/trailing
                    whitespace and converted to lower case.
            
            Returns:
                (bool):
                     True if the input string is one of the truthful values, False if it is
                    one of the false values.
            
            Raises:
                TypeError:
                     If the input string does not match any of the specified truthful
                    or false values, after being processed.

            """
            result = x.strip().lower()
            if result in ["true", "1", "t"]:
                return True
            if result in ["false", "0", "f"]:
                return False
            raise TypeError(x)

        try:
            if self._type is bool:
                return converter(value)
            return self._type(value)
        except (TypeError, ValueError) as e:
            if default is not None:
                return default
            raise LLMParsingException(f"Value {value} cannot be parsed into {self._type.__name__}") from e


class LLMAnswer:
    """
    Class to represent and handle the LLMAnswer object based on a given schema.
    This class provides methods to handle constructing an LLMAnswer object, parsing data from prompts, and validates
    the parsed data based on the schema. It is designed to work with different formats of data input/output such as
    YAML and plain text, handling serialization and deserialization as necessary.
    
    Attributes:
        _schema (Any):
             The schema of the class being represented.
        _class_name (str):
             The name of the schema class.
        _valid_func (Callable):
             An optional function that validates the schema after parsing.
        _properties (List[LLMProperty]):
             A list of properties associated with the schema.
    
    Methods:
        __init__(self, schema:
             Any): Initializes the LLMAnswer object with the given schema.
        field_separator(cls) -> str:
             Class method that returns the string used to separate fields.
        to_prompt(self) -> str:
             Returns a string representation of the properties for a prompt.
        to_yaml_prompt(self) -> str:
             Generates a YAML formatted prompt template.
        to_object(self, line:
             str) -> Optional[Any]: Creates an object from a line of string input.
        parse_line(self, line:
             str, default: Optional[Any]) -> Dict[str, Any]: Parses a line.
        parse_yaml(self, bloc:
             str) -> Dict[str, Any]: Parses a YAML block into a dictionary.
        parse_yaml_list(self, bloc:
             str) -> List[Dict[str, Any]]: Parses a list of YAML blocks.
        parse_yaml_bloc(self, bloc:
             str) -> Dict[str, Any]: Parses a block of text with embedded YAML code.
        _find(self, prop:
             str) -> Optional[LLMProperty]: Finds and returns the LLMProperty associated with the given name.
        

    """
    def __init__(self, schema: Any):
        """
        Initializes a new instance of the class which works as a parser for a given schema.
        This constructor inspects the provided schema to discover `llm_property` attributes and
        `llm_class_validator` methods. It collects information about each property,
        such as its name, type, description, and rank. These properties are then sorted based on their rank.
        If a class validator method is found, it is stored for future validation calls.
        
        Args:
            schema (Any):
                 The class schema from which to derive properties and validation method.
        
        Raises:
            LLMParsingException:
                 If the schema does not have the expected format or content.

        """
        self._schema = schema
        self._class_name = schema.__name__
        self._valid_func = None
        properties = []
        getmembers = inspect.getmembers(schema)
        for attr_name, attr_value in getmembers:
            if isinstance(attr_value, llm_property):
                prop_info = LLMProperty(name=attr_name, prop=attr_value)
                properties.append(prop_info)
            if isinstance(attr_value, llm_class_validator):
                self._valid_func = attr_value.f
        properties.sort(key=lambda item: item.rank)
        self._properties = properties

    @staticmethod
    def field_separator() -> str:
        """
        
        Returns a specific string that is used as a separator between fields.
        
        Returns:
            (str):
                 A string representing the field separator.

        """
        return "<->"

    def to_prompt(self) -> str:
        """
        Generates a prompt string from the object's properties.
        This method constructs a prompt string by concatenating the object's properties using the separator obtained from `field_separator()`. Each property is converted to a string and then these strings are joined together.
        
        Returns:
            (str):
                 A string containing all the properties of the object, separated by the field separator.
            

        """
        p = [f"{prop}" for prop in self._properties]
        return self.field_separator().join(p)

    def to_yaml_prompt(self) -> str:
        """
        Generates a YAML formatted prompt string based on the properties of the current class.
        This method constructs a string that serves as a prompt for users to provide their
        answers in a YAML format. The prompt includes a template that specifies how the
        YAML should be structured, mentioning the class name and listing all available
        properties of the class for which the YAML is to be provided. The properties are
        indented to align with the YAML formatting standards.
        
        Returns:
            (str):
                 A string representing the YAML prompt template, including the class name
                and its properties, formatted according to YAML standards.

        """
        fp = [
            "Use precisely the following template:",
            "```yaml",
            f"your yaml formatted answer for the `{self._class_name}` class.",
            "```",
            "\n",
        ]
        p = [f"  {prop}" for prop in self._properties]
        return "\n".join(fp) + self._class_name + ":\n" + "\n".join(p) + "\n"

    def to_object(self, line: str) -> Optional[Any]:
        """
        Parses a line of text and attempts to convert it into an object of a predefined schema.
        This method takes a string representation of a line, parses it according to the parser's logic, and constructs an object based upon a specified schema. It checks if all required properties are present within the parsed line. If any required properties are missing, it raises an LLMParsingException with an informative message about the missing keys. If a validation function is set, it will also ensure that the created object meets certain criteria determined by that function.
        
        Args:
            line (str):
                 The line of text to be parsed and converted into an object.
        
        Returns:
            (Optional[Any]):
                 An object of the predefined schema created from the parsed line, or None if the line doesn't contain necessary information.
        
        Raises:
            LLMParsingException:
                 If the parsed line is missing any required properties.

        """
        d = self.parse_line(line, None)
        missing_keys = [key.name for key in self._properties if key.name not in d.keys()]
        if len(missing_keys) > 0:
            raise LLMParsingException(f"Missing `{missing_keys}` in response.")
        t = self._schema(**d)
        if self._valid_func is not None:
            self._valid_func(t)
        return t

    def parse_line(self, line: str, default: Optional[Any] = "Invalid") -> Dict[str, Any]:
        """
        Parses a line of text to extract properties and their values into a dictionary.
        This function takes a line of text, which is expected to contain property-value pairs separated by
        a defined field separator. Each pair is further split by a colon to isolate the property name from
        the property value. Property names undergo cleansing to remove unwanted characters,
        such as apostrophes and hyphens, and to trim whitespace. Each property name is validated
        against the class properties using the '_find' method. If a valid property is identified,
        the associated value is parsed and converted into the appropriate data type using the
        property's parse method. The parsed value is then stored in a dictionary with the property
        name as the key.
        If a property cannot be parsed, a provided default value is used instead.
        If the line contains text that does not conform to the expected pair format, it is ignored.
        
        Args:
            line (str):
                 A string representing the line to be parsed.
            default (Optional[Any]):
                 A default value to use when parsing fails. Defaults to 'Invalid'.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing parsed property and value pairs.
            

        """
        property_value_pairs = line.split(self.field_separator())
        properties_dict = {}
        for pair in property_value_pairs:
            if ":" not in pair:
                continue
            values = pair.split(":", 1)
            prop_name = values[0].replace("'", "")
            prop_name = prop_name.replace("-", "")
            prop_name = prop_name.strip()
            prop_value = values[1].strip()

            class_prop = self._find(prop_name)
            if class_prop is not None:
                typed_value = class_prop.parse(prop_value, default)
                properties_dict[class_prop.name] = typed_value
        return properties_dict

    def parse_yaml(self, bloc: str) -> Dict[str, Any]:
        """
        Parses a YAML string into a dictionary structure and validates the presence of expected keys defined in `_properties`.
        This function loads a YAML string (bloc), converts it to a dictionary, and then checks whether all keys defined
        in the instance attribute `_properties` are present. If any keys are missing, it raises an LLMParsingException.
        
        Args:
            bloc (str):
                 A string containing the YAML configuration.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing the keys and values parsed from the YAML string.
        
        Raises:
            LLMParsingException:
                 If any keys defined in `_properties` are missing from the parsed dictionary.
            

        """
        d = yaml.safe_load(bloc)
        properties_dict = {**d}
        missing_keys = [key.name for key in self._properties if key.name not in properties_dict.keys()]
        if len(missing_keys) > 0:
            raise LLMParsingException(f"Missing `{missing_keys}` in response.")
        return properties_dict

    def parse_yaml_list(self, bloc: str) -> List[Dict[str, Any]]:
        """
        Parses a YAML formatted string into a list of dictionaries with specific properties.
        This function loads a YAML formatted string (`bloc`) and converts each item
        in the YAML list into a dictionary, which is then validated to ensure that
        it contains all the required keys as specified by the `_properties` of the
        object. If any required keys are missing, an LLMParsingException is raised
        with a message stating the missing keys.
        
        Args:
            bloc (str):
                 A string containing the YAML formatted list to be parsed.
        
        Returns:
            (List[Dict[str, Any]]):
                 A list of dictionaries, each representing one item
                in the YAML list with the appropriate properties populated.
        
        Raises:
            LLMParsingException:
                 If any item in the YAML list is missing required keys.
            

        """
        result = []
        d = yaml.safe_load(bloc)
        for item in d:
            properties_dict = {**item}
            missing_keys = [key.name for key in self._properties if key.name not in properties_dict.keys()]
            if len(missing_keys) > 0:
                raise LLMParsingException(f"Missing `{missing_keys}` in response.")
            result.append(properties_dict)
        return result

    def parse_yaml_bloc(self, bloc: str) -> Dict[str, Any]:
        """
        Parses the given YAML block and returns a dictionary representation of it.
        The method searches for a YAML code block within the provided text block,
        parses it if found, and then converts the parsed YAML information into a dictionary.
        
        Args:
            bloc (str):
                 The text block potentially containing a YAML code block.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary representation of the parsed YAML block.
                If no YAML code block is found within the text, an empty dictionary is returned.
            

        """
        code_bloc = CodeParser.find_first(language="yaml", text=bloc)
        if code_bloc is not None:
            return self.parse_yaml(code_bloc.code)
        return {}

    def _find(self, prop: str) -> Optional[LLMProperty]:
        """
        Searches for a property with a matching name within the instance's property list.
        This method iterates through the list of properties held by the instance and performs a case-insensitive
        match against the provided property name. If a matching property is found, it is returned; otherwise, None is returned.
        
        Args:
            prop (str):
                 The name of the property to find, case insensitive.
        
        Returns:
            (Optional[LLMProperty]):
                 The matching LLMProperty object if found, otherwise None.
            

        """
        for p in self._properties:
            if p.name.casefold() == prop.casefold():
                return p
        return None
