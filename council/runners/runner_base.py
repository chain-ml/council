"""

Module `runner_base` defines the base class for runners which manage execution logic, error handling, and timeouts within a given context.

Classes:
    RunnerBase(Monitorable, abc.ABC): An abstract base class that sets up the execution framework for runners.
    
Functions:
    run_from_chain_context(context: ChainContext, executor: RunnerExecutor) -> None
        Calls the `run` method to start execution within the given context.
    
    fork_run_merge(runner: Monitored[RunnerBase], context: ChainContext, executor: RunnerExecutor)
        Creates a fork of the given context, executes the run method, and then merges the context back.
    
    run(context: ChainContext, executor: RunnerExecutor) -> None
        The main execution method that all subclasses need to implement, handling execution flow and errors.
    
    rethrow_if_exception(fs: Set[futures.Future])
        Checks a set of `Future` instances for exceptions and rethrows them immediately.
    
    _run(context: ChainContext, executor: RunnerExecutor) -> None
        Abstract method to be implemented by subclasses to define the runner's execution logic.

Exceptions:
    RunnerError(Exception):
        A base exception for runner-specific errors.
    
    RunnerTimeoutError(RunnerError):
        An exception to be raised when a runner encounters a timeout.


"""
from __future__ import annotations

import abc
from collections.abc import Set
from concurrent import futures

from council.contexts import ChainContext, Monitorable, Monitored
from .errrors import RunnerError, RunnerTimeoutError
from .runner_executor import RunnerExecutor


class RunnerBase(Monitorable, abc.ABC):
    """
    A base class implementing the execution logic for runners within a monitored context, adhering to the
    abstract base class (ABC) interface. This class is designed to be subclassed to create specific runner implementations
    and is not intended to be instantiated directly. With methods for managing execution flow, error handling, and context
    management, it provides a scaffold for its subclasses to define their specific run logic by implementing the
    _run abstract method.
    
    Attributes:
        Inherits all attributes from Monitorable and abc.ABC.
    
    Methods:
        run_from_chain_context(context, executor):
            Directly invokes the run method, providing a means to execute the runner from a chain context.
        fork_run_merge(runner, context, executor):
            Forks the provided context for a specified runner, executes the inner run method of the runner, and finally
            merges the context back.
        run(context, executor):
            Orchestrates the running process for a runner, checking if the context should stop the execution, handling context
            entering and exiting, executing the runner's run logic, and handling exceptions such as timeouts and unexpected
            errors.
        rethrow_if_exception(fs):
            Static method that iterates over a set of futures to check for exceptions, rethrowing any exceptions encountered.
        _run(context, executor):
            An abstract method that subclasses must implement to define their specific execution logic within the provided
            context and using the designated executor.

    """
    def run_from_chain_context(self, context: ChainContext, executor: RunnerExecutor) -> None:
        """
        Runs the current instance's `run` method using the supplied chain context and executor.
        
        Args:
            context (ChainContext):
                 The context associated with the chain of execution.
            executor (RunnerExecutor):
                 The executor that facilitates execution.
        
        Returns:
            (None):
                 This method does not return anything.
        
        Raises:
            Any exceptions raised are not captured within this method, and if the `run` method
            of the current instance throws any exceptions, they will propagate outward.

        """
        self.run(context, executor)

    """
    Base runner class that handles common execution logic, including error management and timeout
    """

    def fork_run_merge(self, runner: Monitored[RunnerBase], context: ChainContext, executor: RunnerExecutor):
        """
        Performs a forking operation on the given ChainContext to allow a RunnerBase's 'inner' run method to be executed in isolation, and then merges the resulting context back into the original ChainContext.
        This function first forks a new ChainContext from the provided 'context' using the 'runner' as the monitored entity. It then executes the runner's 'inner' run method using the newly created ChainContext 'inner' and an executor. After the execution is complete, irrespective of its success or failure, the function will attempt to merge the 'inner' context back into the original 'context'.
        
        Args:
            runner (Monitored[RunnerBase]):
                 The runner instance carrying the action to be executed in isolation.
            context (ChainContext):
                 The original ChainContext from which a new context will be forked for isolated execution.
            executor (RunnerExecutor):
                 The executor that will run the 'inner' method of the runner.
        
        Returns:
            None
        
        Raises:
            Any exception raised by the runner's 'inner' run method or the forking process is not explicitly caught within this function, thus the caller needs to handle exceptions.

        """
        inner = context.fork_for(runner)
        try:
            runner.inner.run(inner, executor)
        finally:
            context.merge([inner])

    def run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Determines whether the execution should be stopped based on certain conditions.
        Checks if the budget associated with the execution is expired or if the cancellation token has been set. Logs the reason for stopping if either condition is met.
        
        Returns:
            (bool):
                 True if the execution should be stopped, otherwise False.

        """
        if context.should_stop():
            return

        context.logger.debug("start running %s", self.__class__.__name__)
        try:
            with context:
                self._run(context, executor)
        except futures.TimeoutError as e:
            context.logger.debug("timeout running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerTimeoutError(self.__class__.__name__) from e
        except RunnerError:
            context.logger.debug("runner error running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise
        except Exception as e:
            context.logger.exception("an unexpected error occurred running %s", self.__class__.__name__)
            context.cancellation_token.cancel()
            raise RunnerError(f"an unexpected error occurred in {self.__class__.__name__}") from e
        finally:
            context.logger.debug("done running %s", self.__class__.__name__)

    @staticmethod
    def rethrow_if_exception(fs: Set[futures.Future]):
        """
        
        Raises any exceptions that occurred in the set of futures provided.
            This static method takes a set of `concurrent.futures.Future` objects and calls `result` on each,
            forcing any exceptions that occurred during their execution to be re-raised in the current execution context.
        
        Args:
            fs (Set[futures.Future]):
                 A set of `concurrent.futures.Future` objects.
        
        Raises:
            concurrent.futures.TimeoutError:
                 If a timeout occurs while waiting for the future's result.
            Exception:
                 Any exception that was raised during the execution of the future.

        """
        [f.result(timeout=0) for f in fs]

    @abc.abstractmethod
    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Runs a task using the provided execution context and executor.
        This abstract method must be implemented by subclasses to define the execution of a task within a specific context using the given executor. This method is intended to be called internally by the task execution framework, and it outlines the steps necessary to run a task. The actual logic of task execution will be provided in the concrete implementations of this method.
        
        Args:
            context (ChainContext):
                 The context in which the task is to be executed. This object contains information and the state necessary for running the task, such as data inputs, environmental variables, and other relevant context.
            executor (RunnerExecutor):
                 The executor responsible for running the task. This is the mechanism through which the task's execution is managed and coordinated.
        
        Returns:
            (None):
                 This method does not return any value and is expected to either complete successfully or raise an exception in case of errors.
        
        Raises:
            NotImplementedError:
                 If this method is not implemented in a subclass, calling it will raise a NotImplementedError, as it is an abstract method.

        """
        pass
