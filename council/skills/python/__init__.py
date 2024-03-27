"""

Module __init__.

This module contains the required imports and specialized skills for a coding environment focusing
on Python code generation, verification, and execution. The skills are designed to interact within
a council context, allowing iterative improvement and execution of Python code within predefined
workflows.

Functions:
    build_code_generation_loop(code_generation, verification=None, execution=None, max_iteration=10) -- Construct a loop to run Python code generation,
        optionally followed by verification and execution, with a maximum number of iterations.

Attributes:
    itertools (module): Provides access to iterator building functions.
    List (type): Type indicating a list of elements.
    Optional (type): Type indicating an optional element.
    PythonCodeGenerationSkill (class): Skill for generating Python code based on given parameters.
    PythonCodeVerificationSkill (class): Skill to verify the correctness of Python code against given criteria.
    PythonCodeExecutionSkill (class): Skill to execute Python code within a managed environment.
    ChainContext (class): Contextual object representing the current state of an execution chain within a council framework.
    RunnerBase (class): Base class for runners used to handle execution logic in a council context.
    DoWhile (class): Runner that executes a workflow in a do-while loop behavior based on a provided predicate.
    If (class): Conditional runner that executes one workflow based on a predicate's outcome.
    Sequential (class): Runner that manages the sequential execution of multiple workflows.



"""
import itertools
from typing import List, Optional

from .python_code_generation_skill import PythonCodeGenerationSkill
from .python_code_verification_skill import PythonCodeVerificationSkill
from .python_code_execution_skill import PythonCodeExecutionSkill
from council.contexts import ChainContext
from council.runners import RunnerBase, DoWhile, If, Sequential


def build_code_generation_loop(
    code_generation: PythonCodeGenerationSkill,
    verification: Optional[PythonCodeVerificationSkill] = None,
    execution: Optional[PythonCodeExecutionSkill] = None,
    max_iteration: int = 10,
) -> RunnerBase:
    """
    Generates a loop for building, verifying, and executing code using specified skills with a maximum iteration limit.
    This function constructs a loop that executes a code generation skill. If verification and execution skills are provided,
    they are conditionally included in the loop. The loop iterates until the predicate condition fails, an error occurs, or
    the maximum number of iterations is reached.
    Arguments:
    code_generation (PythonCodeGenerationSkill): The skill responsible for code generation.
    verification (Optional[PythonCodeVerificationSkill]): The skill used for verifying generated code. Defaults to None.
    execution (Optional[PythonCodeExecutionSkill]): The skill for executing the generated code. Defaults to None.
    max_iteration (int): The maximum number of iterations to execute the loop before stopping.
    
    Returns:
        (RunnerBase):
             A DoWhile loop runner that encapsulates the code generation, verification, and execution process.
    
    Raises:
        RunnerPredicateError:
             An error occurred while evaluating the loop's predicate condition.

    """

    def while_predicate(context: ChainContext) -> bool:
        """
        Checks whether the chain should continue its execution based on the presence of error messages and a maximum iteration threshold.
        
        Args:
            context (ChainContext):
                 The context within which the chain is operating. This includes messages that may contain errors.
        
        Returns:
            (bool):
                 A boolean value indicating whether the chain execution should continue. It returns True if the last message is an error and the number of encountered
                error messages has not yet reached the maximum allowed iteration count (`max_iteration`). Otherwise, it returns False.
        
        Note:
            This function assumes that the `max_iteration` variable is defined in the calling scope.

        """
        error_messages = filter(lambda m: m.is_error, context.messages)
        count = sum(1 for _ in itertools.islice(error_messages, max_iteration))
        last_message = context.last_message
        return last_message is not None and last_message.is_error and count < max_iteration

    def is_ok(context: ChainContext) -> bool:
        """
        Checks if the last message in the given context is not None and is flagged as okay.
        This function evaluates the state of the `last_message` attribute within the provided `ChainContext` object,
        returning True if the `last_message` exists and its `is_ok` property is set to True. It returns False otherwise.
        
        Args:
            context (ChainContext):
                 An object representing the chain context which includes details about the last message.
        
        Returns:
            (bool):
                 A boolean indicating whether the last message is not None and its `is_ok` flag is True.

        """
        return context.last_message is not None and context.last_message.is_ok

    sequence: List[RunnerBase] = [code_generation]
    if verification is not None:
        sequence.append(If(is_ok, verification))
    if execution is not None:
        sequence.append(If(is_ok, execution))

    return DoWhile(while_predicate, Sequential(*sequence))
