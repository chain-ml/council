"""

Module-level docstring for '__init__' module.

This module is the initializer that imports and exposes the core classes related to the execution
unit and controller components of the system. This includes the ExecutionUnit class which represents
a single execution context for operations, the ControllerException class which serves as the
custom exception for controller-related errors, ControllerBase class that serves as the
abstract base class for all controllers, the BasicController class which extends the ControllerBase
class providing a simple execution framework and LLMController which extends ControllerBase
to provide specific functionalities for language model-based execution control.

Classes:
    ExecutionUnit: Represents a single executable context with associated metadata.
    ControllerException: Custom exception class for handling controller-related errors.
    ControllerBase: Abstract base class for all controllers defining the execution workflow interface.
    BasicController: A simple concrete implementation of ControllerBase.
    LLMController: An advanced controller utilizing a language model for decision-making.


"""

from .execution_unit import ExecutionUnit
from .controller_base import ControllerException, ControllerBase
from .basic_controller import BasicController
from .llm_controller import LLMController
