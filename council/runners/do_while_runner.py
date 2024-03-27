"""

Module do_while_runner

This module provides an implementation of a 'do-while' control flow mechanism within a runner execution
context using `Council`, a framework for managing complex workflows. It defines the `DoWhile` class that
inherits from `RunnerBase` and encapsulates a looping structure where a body runner is executed at least once
and continues execution as long as a given predicate is True.

Classes:
    DoWhile: A `RunnerBase` subclass that implements the do-while loop logic.

Usage:
    Instantiate `DoWhile` with a `RunnerPredicate` and a `RunnerBase` instance. The `RunnerPredicate` is a
    callable that takes a `ChainContext` and returns a boolean, determining whether the loop will continue
    to run. The body runner, provided as `RunnerBase`, defines the actions to perform in each iteration of
    the loop. Call the `_run` method passing in a `ChainContext` and a `RunnerExecutor` to execute the
    do-while loop.

Exceptions:
    RunnerPredicateError: An exception raised when the predicate evaluation results in an exception.



"""
from council.contexts import ChainContext, ChatMessage
from council.runners import RunnerBase, RunnerPredicate, RunnerExecutor, RunnerPredicateError


class DoWhile(RunnerBase):
    """
    A specialized RunnerBase subclass that executes a runner in a do-while loop construct.
    This class models a traditional do-while loop structure where a particular piece of
    logic (runner) is executed at least once, and then repeatedly as long as a given
    predicate condition is true.
    
    Attributes:
        _predicate (RunnerPredicate):
             A callable predicate representing the loop
            continuation condition. It takes a single argument, a ChainContext,
            and returns a boolean.
        _body (RunnerMonitor):
             A RunnerMonitor initialized with the runner to be
            executed within the do-while loop. It monitors the execution of the runner.
    
    Methods:
        __init__:
             Initializes the DoWhile instance, setting up the predicate and body runner.
        _run:
             Executes the runner within the loop and continues the loop as long
            as the predicate evaluates to True.
        check_predicate:
             Evaluates the predicate with the provided context and handles
            any exceptions raised during its execution.
    
    Raises:
        RunnerPredicateError:
             If the predicate raises an exception during execution.

    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        """
        Initializes a new instance of the doWhileRunner class.
        This constructor initializes the doWhileRunner with a given predicate and runner, setting
        up a monitor for the runner's execution inside a do-while construct. It inherits from a parent
        class (implicitly) and sets the instance's name to 'doWhileRunner'.
        
        Args:
            predicate (RunnerPredicate):
                 An instance of RunnerPredicate that defines the condition to be checked
                before each execution of the runner.
            runner (RunnerBase):
                 An instance of RunnerBase or any of its subclasses that represents the
                code to be executed repeatedly as long as the predicate condition is true.
            

        """
        super().__init__("doWhileRunner")
        self._predicate = predicate
        self._body = self.new_monitor("doWhileBody", runner)

    def _run(self, context: ChainContext, executor: RunnerExecutor) -> None:
        """
        Executes the main loop of the component's logic in a given context using a specified executor until a condition is no longer met.
        This method encapsulates what could be considered the "heart" of the component's execution process. It continuously invokes a defined inner logic (or body) to run, provided by the '_body.inner.run' method, within a context and using an executor for task execution. The loop is controlled by a predicate function 'check_predicate', which evaluates the context to determine if the loop should continue or exit.
        
        Args:
            context (ChainContext):
                 The context in which the execution occurs. This should be an instance of ChainContext or its subclass, which provides the necessary state and interfaces for the execution.
            executor (RunnerExecutor):
                 An executor instance responsible for running the tasks. It should comply with the RunnerExecutor interface. This parameter is used to carry out individual tasks that the '_body.inner.run' method might issue.
        
        Returns:
            (None):
                 This method does not return any value and is solely responsible for performing its loop-based execution logic.
        
        Raises:
            Any exceptions raised would depend on the implementation specifics of the '_body.inner.run' method and the 'check_predicate' condition. This method itself does not explicitly raise any exceptions.

        """
        while True:
            self._body.inner.run(context, executor)

            if not self.check_predicate(context):
                return

    def check_predicate(self, context: ChainContext) -> bool:
        """
        Determines whether a specified condition is met within the given context.
        This method evaluates a predicate function that takes a `ChainContext` as an argument. If the predicate function executes successfully,
        it returns a boolean result. If an exception occurs during the execution of the predicate function, an error message is
        appended to the context, and a `RunnerPredicateError` is raised with the original exception attached.
        
        Args:
            context (ChainContext):
                 The context in which the predicate will be evaluated.
        
        Returns:
            (bool):
                 True if the predicate is successful, otherwise False.
        
        Raises:
            RunnerPredicateError:
                 An error wrapping the original exception to indicate that an issue occurred while
                evaluating the predicate.
            

        """
        try:
            return self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("DoWhileRunner", f"Predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e
