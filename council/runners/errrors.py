"""

Module for custom exceptions used within a Runner context.

This module defines a set of custom exception classes that are intended to be used to
handle various types of errors that may occur during the execution of a Runner - an
abstract entity that could represent a task executor or a workflow controller.

Classes:
    RunnerError(Exception): Base class for all Runner-related exceptions.
    RunnerTimeoutError(RunnerError): Raised when a Runner operation exceeds its allocated time limit.
    RunnerSkillError(RunnerError): Signifies an error related to the skill or competence required by the Runner.
    RunnerPredicateError(RunnerError): Indicates an error where a certain predicate or condition has not been met during Runner execution.
    RunnerGeneratorError(RunnerError): Raised when an error is encountered in a generator used by the Runner.


"""
class RunnerError(Exception):
    """
    Custom exception class that inherits from Python's built-in Exception class.
    This custom exception class `RunnerError` is intended to be used to signal issues
    related to a 'Runner' entity when performing operations where standard exceptions
    are not suitable. More specific context can be provided when raising this exception
    by passing an appropriate error message to the constructor.
    
    Attributes:
        Inherits all attributes from the superclass `Exception`.
    
    Methods:
        Inherits all methods from the superclass `Exception`, including the constructor,
        which can take an optional error message parameter.
        

    """

    pass


class RunnerTimeoutError(RunnerError):
    """
    A custom exception class that inherits from RunnerError, indicating that a timeout has occurred within a runner process.
    
    Attributes:
        None
    
    Methods:
        None

    """

    pass


class RunnerSkillError(RunnerError):
    """
    Custom exception class to represent errors related to a Runner's skills. Inherits from RunnerError.
    
    Attributes:
        Inherits all attributes from the parent class RunnerError.
    
    Methods:
        Inherits all methods from the parent class RunnerError.
    
    Raises:
        RunnerSkillError:
             An error specifically related to Runner skills.

    """

    pass


class RunnerPredicateError(RunnerError):
    """
    Error raised when a predicate in the Runner fails.
    
    Attributes:
        message (str):
             Human readable string describing the exception.
        code (int, optional):
             Numeric code representing the error.
            This exception is raised when a specific predicate condition within the Runner
            component is not met, indicating that an operation cannot proceed due to
            not satisfying preconditions or invariants expected by the Runner logic.
        Inherits from:
        RunnerError:
             A generic error class for Runner-related exceptions.

    """

    pass


class RunnerGeneratorError(RunnerError):
    """
    RunnerGeneratorError class.
    This class is a specialized error that extends RunnerError. It is
    raised when an error specific to the runner's generator operation
    occurs. This could involve issues related to generation processes
    within a runner context, such as when tasks or workflows fail to
    generate as expected.
    
    Attributes:
        Inherits all attributes from the parent class RunnerError.
        

    """

    pass
