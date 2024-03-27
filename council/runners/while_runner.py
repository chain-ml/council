"""


Module while_runner
--------------------

This module contains the implementation of a `While` class which inherits from `RunnerBase`.
The `While` class is designed to run a given procedure repeatedly while a specified predicate condition is true.

Classes:
    While: A runner that repeatedly executes a body runner as long as the given predicate holds true.

Exceptions:
    RunnerPredicateError: An error thrown when the runner predicate encounters an exception.



"""
from council.contexts import ChainContext, ChatMessage
from council.runners import RunnerBase, RunnerPredicate, RunnerExecutor, RunnerPredicateError


class While(RunnerBase):
    """
    Class While inherits from RunnerBase and provides functionality to execute a body of code repeatedly as long as a given predicate is true.
    
    Attributes:
        _predicate (RunnerPredicate):
             The predicate to evaluate before each execution of the runner.
        _body (RunnerMonitor):
             A monitor wrapper for the runner that will be executed repeatedly as long as the predicate evaluates to True.
    
    Methods:
        __init__(self, predicate:
             RunnerPredicate, runner: RunnerBase):
            Initializes the While class instance by setting the 'whileRunner' label, the predicate and creating a monitor for the runner.
        _run(self, context:
             ChainContext, executor: RunnerExecutor):
            Executes the body runner repetitively until the predicate returns False. Each iteration checks the predicate and if True, executes the runner.
        check_predicate(self, context:
             ChainContext) -> bool:
            Evaluates the predicate with the given context. If the predicate raises an exception, it logs the exception as an error and re-raises a RunnerPredicateError.
    
    Raises:
        RunnerPredicateError:
             An error raised if the predicate raises an exception during evaluation.

    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase):
        """
        Initializes the 'whileRunner' with a specific predicate and runner.
        This instance is responsible for repeatedly executing the provided runner as long as
        the given predicate evaluates to True. Upon initialization, a monitor named 'whileBody' is also created and associated
        with the runner to keep track of its execution.
        
        Args:
            predicate (RunnerPredicate):
                 An object representing the condition to be checked before each execution of the runner.
            runner (RunnerBase):
                 The runner object that will be executed as long as the predicate is True.

        """
        super().__init__("whileRunner")
        self._predicate = predicate
        self._body = self.new_monitor("whileBody", runner)

    def _run(self, context: ChainContext, executor: RunnerExecutor):
        """
        Executes the loop body if the predicate condition is satisfied.
        This method implements a loop mechanism that repeatedly runs the body of the construct while the condition checked by check_predicate remains True. The execution uses provided context and executor to run the loop body defined by the _body.inner attribute.
        
        Args:
            context (ChainContext):
                 The context in which the loop is being executed. This includes any runtime state or data that should be passed along to the loop body during iteration.
            executor (RunnerExecutor):
                 An executor that is responsible for running the command(s) specified in the loop body. The executor provides the necessary interface to execute the body correctly within the given context.
        
        Raises:
            Any exception that might be raised during the execution of the loop body or predicate check will propagate upwards, as no specific error handling is performed by this method.
            

        """
        while self.check_predicate(context):
            self._body.inner.run(context, executor)

    def check_predicate(self, context: ChainContext) -> bool:
        """
        Determines whether the predicate condition is met within the provided context.
        This method is responsible for evaluating a predicate function that takes a `ChainContext` object as its argument. If the predicate function executes without any problems and returns `True`, it implies that the condition is met, and the method will return `True`. On the other hand, if the predicate returns `False`, the method will return `False`. In the case where executing the predicate function raises an exception, the method logs an error message using `ChatMessage.skill` and raises a `RunnerPredicateError` with the original exception as the cause.
        
        Args:
            context (ChainContext):
                 The context in which the predicate will be checked. This object provides all the necessary information that the predicate might require to evaluate its condition.
        
        Returns:
            (bool):
                 A boolean value indicating whether the predicate condition is met (`True`) or not (`True`).
        
        Raises:
            RunnerPredicateError:
                 An error indicating that there was an issue while trying to evaluate the predicate. This is raised with the original exception that caused the failure as its cause.
            

        """
        try:
            return self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("WhileRunner", f"Predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e
