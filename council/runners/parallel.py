"""

A module for running tasks in parallel using a pool of runners derived from RunnerBase.

This module defines the Parallel class which inherits from RunnerBase and is used to execute
multiple RunnerBase instances concurrently. The main aim is to leverage concurrency to improve
performance when running tasks that can be executed in parallel.

Attributes:
    _runners (List[Monitored[RunnerBase]]): A list of monitored runners that will be executed in parallel.

Methods:
    __init__(self, *runners: RunnerBase):
        Initializes a new instance of the Parallel class with the provided runners.

    _run(self, context: ChainContext, executor: RunnerExecutor):
        Implements the abstract method from RunnerBase to execute the runners
        in parallel. It forks the ChainContext for each runner, submits them for
        execution, and waits for either all futures to complete or the first exception to be raised.
        It ensures that contexts are merged and futures are cancelled properly.

Raises:
    RunnerTimeoutError: If the execution exceeds the allocated time budget.
    RunnerError: If any runner-specific error occurs during execution.
    Exception: If an unexpected error occurs during the parallel execution process.


"""
from concurrent import futures
from council.contexts import ChainContext

from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor


class Parallel(RunnerBase):
    """
    Class for executing multiple RunnerBase instances in parallel.
    This class is responsible for managing parallel execution of runners, each derived from RunnerBase.
    It manages their lifecycle, runs them utilizing a given executor, and handles proper termination and
    aggregation of execution contexts.
    
    Attributes:
        _runners (List[RunnerBase]):
             A list of runners that will execute in parallel.
    
    Args:
        *runners (RunnerBase):
             An unspecified number of RunnerBase instances to be executed in parallel.
    
    Methods:
        __init__:
             Constructor to initialize a Parallel instance with one or more runners.
        _run:
             Orchestrates the parallel execution of runners using the given context and executor.
    
    Note:
        The _run method is intended to be used internally and should not be called directly.

    """

    def __init__(self, *runners: RunnerBase):
        """
        Initializes a new instance of the class with specified runners.
        This special initializer for a class will create a new object that is capable of handling multiple
        'RunnerBase' instances in parallel. It uses a unique identifier 'parallelRunner' for the created object
        and sets up monitoring for the runners provided as arguments.
        
        Args:
            *runners (RunnerBase):
                 Variable length argument list of RunnerBase instances.
        
        Raises:
            TypeError:
                 If any argument provided is not an instance of RunnerBase.
            

        """
        super().__init__("parallelRunner")
        self._runners = self.new_monitors("parallel", runners)

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Runs the specified runners with a forked context in parallel using the provided executor and handles their completion.
        This internal method schedules the execution of each runner with its forked context in the executor,
        waits for either all futures to complete or the first exception to occur within the budget's
        remaining duration, and then merges the messages from each context back into the main context.
        If an exception occurs in any of the runners, it is rethrown to the caller. Futures are cancelled
        if the try block is exited for any reason, ensuring no lingering executions.
        
        Args:
            context (ChainContext):
                 The main context from which to fork.
            executor (RunnerExecutor):
                 The executor used to run runners in parallel.
        
        Raises:
            Any exception that occurs within the runner's execution is caught and rethrown.

        """
        contexts = [(runner.inner, context.fork_for(runner)) for runner in self._runners]

        # Seems like it is a bad idea using lambda as the function in submit,
        # which results into inconsistent invocation (wrong arguments)
        fs = [executor.submit(runner.run, inner, executor) for (runner, inner) in contexts]
        try:
            dones, not_dones = futures.wait(fs, context.budget.remaining_duration, futures.FIRST_EXCEPTION)
            self.rethrow_if_exception(dones)
        finally:
            context.merge([context for (_, context) in contexts])
            [f.cancel() for f in fs]
