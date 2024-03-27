"""

A module initializer for a collection of classes and functions supporting environmentally-aware application configuration,
result handling, and data object mapping.

This module provides a set of utilities that are often needed for working with environment variables,
mapping configurations to data objects, and handling the results of operations in a functional
programming style. Through the provided classes and functions, applications can perform safe access
and manipulation of environment variables, define parameters with built-in validation and conversion,
express complex operations' results with Ok/Err semantics, and define structured data objects.

Imports:
    - Option and OptionException from the `option` module for optional value handling.
    - read_env_str, read_env_int, read_env_bool, read_env_float for reading and converting environment variables.
    - MissingEnvVariableException, EnvVariableValueException for environment variable-related exceptions.
    - Ok, Err, Result for result types that can either represent success (Ok) or error (Err).
    - ParameterValueException, Parameter, greater_than_validator, prefix_validator, not_empty_validator
      for defining, validating, and retrieving parameters from the environment.
    - DataObject and DataObjectSpecBase for working with structured data objects.
    - CodeParser for extracting blocks of code from text based on a certain language.
    - OsEnviron for temporarily setting environment variables within a context manager scope.


"""
from .option import Option, OptionException
from .env import (
    read_env_str,
    read_env_int,
    read_env_bool,
    read_env_float,
    MissingEnvVariableException,
    EnvVariableValueException,
)
from .result import Ok, Err, Result
from .parameter import ParameterValueException, Parameter, greater_than_validator, prefix_validator, not_empty_validator
from .data_object import DataObject, DataObjectSpecBase
from .code_parser import CodeParser
from .env import OsEnviron
