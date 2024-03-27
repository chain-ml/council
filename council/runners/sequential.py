"""


Module sequential

This module defines the Sequential class, which inherits from RunnerBase class,
and is used to execute runners in a sequential order.

Classes:
    Sequential(RunnerBase): A class that manages the sequential execution of multiple runners.

The Sequential class provides a constructor to initialize a sequence of runners,
and overrides the _run method from the base class, to execute each runner in the sequence.
It includes a static method from_list to create a Sequential object from a list of runners.

Detailed information on each method within the class is provided below.

    __init__(self, *runners: RunnerBase):
        Initializes a new instance of the Sequential class.
        
        Args:
            *runners: Variable length list of RunnerBase instances representing
                      the runners to be executed in sequence.
    
    _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        Runs each runner in the sequence in the specified context using the given executor.
        Execution halts if the context indicates to stop or an error occurs.
        
        Args:
            context: An instance of ChainContext, which manages the current state
                     of the chain, including logging and error handling.
            executor: An instance of RunnerExecutor, used to execute the runner logic.
    
    from_list(runners: Sequence[RunnerBase]) -> RunnerBase:
        A factory method that creates a Sequential instance from a list of runners.
        If the list has only one runner, it returns that runner instead of creating a Sequential.
        
        Args:
            runners: A sequence of RunnerBase instances to be executed in sequence.
        
        Returns:
            An instance of Sequential or the lone RunnerBase from runners, if there's only one.


"""
from typing import Sequence

from council.contexts import ChainContext
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Sequential(RunnerBase):
    """
    A "Sequential" is a class that extends "RunnerBase", designed to execute
    a sequence of runner objects based on a predefined order. It runs each underlying
    runner object one after the other, providing a mechanism to chain operations
    sequentially in a controlled manner.
    
    Attributes:
        _runners (Sequence[RunnerBase]):
             A sequence of runner instances that
            will be executed in the order they were added.
    
    Methods:
        __init__(self, *runners:
             RunnerBase):
            Initializes a new instance of Sequential with a variable number of
            runner objects.
    
    Args:
        runners (RunnerBase):
             A variable number of runner instances that
            will be executed sequentially.
        _run(self, context:
             ChainContext, executor: RunnerExecutor) -> None:
            Private method that executes each runner in the instance's '_runners'
            attribute, in order. Execution will stop if the context's `should_stop`
            method returns True.
    
    Args:
        context (ChainContext):
             The context object providing environment
            and state for the runners during execution.
        executor (RunnerExecutor):
             The executor instance that manages
            runner execution.
        from_list(runners:
             Sequence[RunnerBase]) -> RunnerBase:
            A static method that creates a new Sequential instance from a
            list of runner objects. If the list only contains one runner
            object, it returns that runner instead of creating a Sequential.
    
    Args:
        runners (Sequence[RunnerBase]):
             An ordered sequence of runner
            instances.
    
    Returns:
        (RunnerBase):
             A Sequential runner if the list has more than
            one runner instance, otherwise the single runner in the list.

    """

    def __init__(self, *runners: RunnerBase):
        """
        Initializes a new instance of a sequence runner.
        This constructor receives one or more runner instances that are stored and monitored
        internally. It calls the superclass' constructor with a predefined name 'sequenceRunner'.
        
        Args:
            *runners (RunnerBase):
                 An unpacked tuple of `RunnerBase` instances which are
                the runners to be included in the sequence. Each argument should be an instance
                of a class that inherits from `RunnerBase`.
            

        """
        super().__init__("sequenceRunner")
        self._runners = self.new_monitors("sequence", runners)

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Runs a sequence of runner objects within a given execution context and using a specified executor.
        This method iterates over the runner objects assigned to the instance. For each runner, it checks if the
        execution context signals to stop the operation. If so, the method returns immediately. Otherwise, it
        continues by forking, running, and merging the results of each runner using the given executor.
        
        Args:
            context (ChainContext):
                 The context that provides runtime information and controls for the current
                chain of operations.
            executor (RunnerExecutor):
                 The executor responsible for managing the execution of runners.
        
        Returns:
            (None):
                 This method does not return any value.

        """
        for runner in self._runners:
            if context.should_stop():
                return

            self.fork_run_merge(runner, context, executor)

    @staticmethod
    def from_list(runners: Sequence[RunnerBase]) -> RunnerBase:
        """
        Create a `Sequential` runner from a list of `RunnerBase` instances.
        This function simplifies the creation of a `Sequential` runner by allowing
        clients to provide an ordered sequence of runners. If a single runner is provided,
        that runner is returned directly without the need to wrap it in a `Sequential`.
        
        Args:
            runners (Sequence[RunnerBase]):
                 A sequence of `RunnerBase` instances.
        
        Returns:
            (RunnerBase):
                 A single `RunnerBase` instance if only one is provided in the list;
                otherwise, a `Sequential` runner containing all the provided runners.
            

        """
        if len(runners) == 1:
            return runners[0]

        return Sequential(*runners)
