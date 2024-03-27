"""

Module that initializes components of a runner framework used for executing tasks in a structured and efficient manner.

This module provides the initialization for a custom execution framework designed to handle running of tasks that may follow various control flows (e.g., sequential, parallel, conditional, looping executions). It defines and imports the necessary errors, types, executor classes, as well as base and specialized runner classes for constructing complex task execution chains.

The module imports and prepares the following classes and functions:
- RunnerError, RunnerTimeoutError, RunnerSkillError, RunnerPredicateError, RunnerGeneratorError: Custom exception classes for error handling within the runner framework.
- RunnerPredicate, RunnerGenerator: Type aliases for predicates and generators used in control flow runners.
- RunnerExecutor, new_runner_executor: A class and factory function for managing task execution in threads.
- RunnerBase: An abstract base class for all runners providing common interface and execution patterns.
- SkillRunnerBase: Abstract base class for skill execution runners that run specific skill-related tasks.
- Sequential: Class that allows running runners in a sequential order.
- Parallel: Class that allows running runners concurrently.
- If: Conditional runner class that executes one runner if a predicate is true, and optionally another if it is false.
- LoopRunnerBase: An abstract base class for loop-related runner classes.
- ParallelFor: Runner class that executes an iterator of tasks in parallel.
- DoWhile: Runner class that repeatedly executes a runner while a predicate returns true.
- While: Runner class that executes a runner as long as a predicate returns true.


"""
from .errrors import RunnerError, RunnerTimeoutError, RunnerSkillError, RunnerPredicateError, RunnerGeneratorError

from .types import RunnerPredicate, RunnerGenerator
from .runner_executor import RunnerExecutor, new_runner_executor
from .runner_base import RunnerBase
from .skill_runner_base import SkillRunnerBase
from .sequential import Sequential
from .parallel import Parallel
from .if_runner import If
from .loop_runner_base import LoopRunnerBase
from .parallel_for import ParallelFor
from .do_while_runner import DoWhile
from .while_runner import While
