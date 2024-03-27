"""

Module that extends the RunnerBase with a LoopRunnerBase class.

This module provides a LoopRunnerBase class which inherits from the RunnerBase class.
The LoopRunnerBase serves as a base class for runners that operate in a looping fashion during execution. It does not introduce additional methods or attributes
beyond those provided by RunnerBase, but it signifies a conceptual specialization where subclasses
can be expected to implement loop-based execution logic.

Basic Usage:
    To create a subclass of LoopRunnerBase, override the abstract methods defined in RunnerBase, ensuring the looping logic is properly handled.

Individual subclasses of LoopRunnerBase must provide implementations for:
    - _run(): The core method that will be called to execute the looping logic.

See Also:
    RunnerBase: The parent class from which LoopRunnerBase inherits its basic structure and functionality.


"""
import abc

from .runner_base import RunnerBase


class LoopRunnerBase(RunnerBase, abc.ABC):
    """
    Class that serves as an abstract base for loop runners.
    Inherits from RunnerBase and implements the abstract base class from 'abc' module. This class is designed to be a base class and should be subclassed to provide specific loop running functionality. It intentionally contains no additional methods or attributes beyond those provided by RunnerBase, serving instead as a placeholder indicating the specialization of RunnerBase into a loop-oriented runner.
    Subclasses are expected to implement the loop-running logic that iterates over a set of tasks or functions repeatedly.
    
    Attributes:
        Inherits all attributes from the RunnerBase class.
    
    Methods:
        Subclasses should implement all abstract methods from the RunnerBase class and may add additional methods for loop specific behavior.

    """

    pass
