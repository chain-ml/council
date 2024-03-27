"""


Module if_runner

This module provides the If implementation which inherits from RunnerBase. It encapsulates logic
for conditional execution of runners based on a specified predicate. If the predicate evaluates
to True, it runs a specified runner; otherwise, it runs an alternate runner if one is provided.
The If class is responsible for executing the given runners depending on the evaluation of the
provided predicate in the context of a running chain. It also handles exceptions that may arise
from the predicate function.

Classes:
    If -- A RunnerBase subclass that performs conditional execution.

Attributes:
    predicate (RunnerPredicate): A callable that determines the branch of execution.
    runner (RunnerBase): The runner to execute if the predicate evaluates to True.
    else_runner (Optional[RunnerBase]): An optional runner to execute if the predicate evaluates to False.

Exceptions:
    RunnerPredicateError: Raised when the predicate function provided to the If runner encounters an exception.


"""
from typing import Optional

from council.contexts import ChatMessage, ChainContext

from .errrors import RunnerPredicateError
from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from .types import RunnerPredicate


class If(RunnerBase):
    """
    A conditional execution runner that branches the execution flow depending on a given predicate.
    This class is used to create a conditional construct within a series of runner executions. It evaluates a predicate condition and, based on the outcome,
    directs the execution to one of the two possible runners. The primary `runner` is invoked if the predicate evaluates to true (the 'then' case),
    and an optional `else_runner` is called if the predicate evaluates to false (the 'else' case).
    
    Attributes:
        _predicate (RunnerPredicate):
             A callable that evaluates to a boolean, determining which runner to execute.
        _then (Monitor):
             A monitor-wrapped runner which is executed if the predicate is true.
        _maybe_else (Optional[Monitor]):
             An optional monitor-wrapped runner which is executed if the predicate is false.
    
    Args:
        predicate (RunnerPredicate):
             The condition to be evaluated before executing the runners.
        runner (RunnerBase):
             The runner to be executed if the predicate evaluates to true.
        else_runner (Optional[RunnerBase]):
             The optional runner to be executed if the predicate evaluates to false.
    
    Raises:
        RunnerPredicateError:
             An error raised if the predicate evaluation raises an exception.

    """

    def __init__(self, predicate: RunnerPredicate, runner: RunnerBase, else_runner: Optional[RunnerBase] = None):
        """
        Initializes an 'ifRunner' instance with a given predicate, runner, and optional else runner.
        This method constructs a conditional execution object that follows an 'if-then-else' structure, where the
        execution between two runners depends on the evaluation of a predicate. If the predicate evaluates to true,
        one runner is executed; otherwise, an optional else runner may be executed if provided.
        
        Args:
            predicate (RunnerPredicate):
                 A predicate instance responsible for evaluating a condition.
            runner (RunnerBase):
                 The runner instance to execute when the predicate evaluates to true.
            else_runner (Optional[RunnerBase]):
                 An optional runner to execute when the predicate evaluates to false.
                Defaults to None, which means no action is taken if the predicate is false.
        
        Notes:
            - The 'ifRunner' type is a utility used for conditional execution within an execution framework.
            - The runners must be instances derived from RunnerBase or implement an equivalent interface.
            Superclass:
                This method calls the superclass' __init__ method with the 'ifRunner' type indicated.

        """
        super().__init__("ifRunner")
        self._predicate = predicate
        self._then = self.new_monitor("then", runner)
        self._maybe_else = self.new_monitor("else", else_runner) if else_runner is not None else None

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Runs the internal functionality of a component, applying a predicate before determining the execution path.
        This method processes the conditional execution logic for the component. It evaluates a predicate function using the provided context. If the predicate returns True, the method triggers the execution of another component referenced as 'then'. If the predicate returns False and an 'else' component is specified, it triggers the execution of the 'else' component. Any exception raised by the predicate is caught, logged as a chat message, and wrapped into a custom exception for further handling.
        
        Args:
            context (ChainContext):
                 The execution context containing state and API access for the chain execution.
            executor (RunnerExecutor):
                 The executor responsible for running the components of the chain.
        
        Raises:
            RunnerPredicateError:
                 An error wrapping an exception raised by the predicate function, indicating an issue during its evaluation.

        """
        try:
            result = self._predicate(context)
        except Exception as e:
            context.append(ChatMessage.skill("IfRunner", f"predicate raised exception: {e}", is_error=True))
            raise RunnerPredicateError from e

        if result:
            self._then.inner.run(context, executor)
        elif self._maybe_else is not None:
            self._maybe_else.inner.run(context, executor)
