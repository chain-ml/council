"""

A module defining the base class for skill runners in a chain execution context.

This module provides a SkillRunnerBase class which extends the functionality of RunnerBase to handle
specific execution flows of skills within the council framework. SkillRunnerBase facilitates
submission and management of skill execution tasks using the chain's execution context and
support for error handling, logging, and timeout management.

Classes:
    SkillRunnerBase: A base class for skill runners implementing the core logic required to run a
        skill and handle potential errors that may occur during execution.

Notes:
    - SkillRunnerBase is an abstract class and requires the implementation of the execute_skill
      method in derived classes.
    - This module also imports necessary components such as SkillContext, IterationContext, ChatMessage,
      ChainContext, and RunnerSkillError, ensuring they are available for use within SkillRunnerBase.
    - The module structure allows for clean extension and custom implementations of skill runners
      for different types of skills within the council framework.


"""
import abc

from council.contexts import SkillContext, IterationContext, ChatMessage, ChainContext
from . import RunnerSkillError

from .runner_base import RunnerBase
from .runner_executor import RunnerExecutor
from ..utils import Option


class SkillRunnerBase(RunnerBase):
    """
    A base class for running skills within an executor providing common structure and behavior for skill execution.
    The SkillRunnerBase class is derived from RunnerBase and focuses on executing and managing skill-related
    actions. It overrides some methods of its parent class to handle skill execution flow, including exception
    handling.
    
    Attributes:
        monitor.name (str):
             The name of the skill to be monitored during its run.
        _name (str):
             The internal name of the skill.
    
    Methods:
        __init__(self, name):
            Initializes a new instance of SkillRunnerBase by setting the monitor's name and the internal
            name of the skill.
        _run(self, context, executor):
            Overrides the parent class's _run method to execute the run_skill method.
        run_skill(self, context, executor):
            Manages the skill's run cycle, submits the task for execution, and applies a timeout based
            on the context's budget.
        run_in_current_thread(self, context, iteration_context):
            Executes the skill in the current thread and manages the skill's context and execution flow.
            Error handling is also addressed within this method.
        execute_skill(self, context):
            An abstract method that needs to be implemented by concrete classes to execute the actual skill.
        from_exception(self, exception):
            Creates a ChatMessage instance indicating an error, using data obtained from the exception.
    
    Raises:
        RunnerSkillError:
             An exception is raised if skill execution encounters an unexpected error.

    """

    def __init__(self, name):
        """
        Initializes an instance of the class with the specified name. This constructor sets the name for the skill monitor and stores the name as an attribute of the instance. It also calls the constructor of the superclass.
        
        Args:
            name (str):
                 The name to be given to the skill monitor and stored as an instance attribute.
            

        """
        super().__init__("skill")
        self.monitor.name = name
        self._name = name

    def _run(
        self,
        context: ChainContext,
        executor: RunnerExecutor,
    ) -> None:
        """
        Executes the skill associated with the current instance within the given context using the specified executor.
        This method serves as a wrapper that delegates the task to the `run_skill` method, allowing for consistent execution behaviour within different instances that are governed by the `_run` method's implementation. It does not return any value, and is designed to operate purely through side effects within the given `context` and `executor`.
        
        Args:
            context (ChainContext):
                 The context in which the skill is executed. It provides the necessary
                environment and state for the execution.
            executor (RunnerExecutor):
                 The executor component responsible for managing and carrying
                out the execution of the skill.
        
        Raises:
            This function does not explicitly raise any exceptions by itself. However, exceptions can be
            raised indirectly if the `run_skill` method, which it calls, encounters any errors during
            its execution.
            

        """
        self.run_skill(context, executor)

    def run_skill(self, context: ChainContext, executor: RunnerExecutor) -> None:
        """
        Executes the skill within the given context and using the specified executor.
        This method submits the skill's `run_in_current_thread` method to the executor and waits for the execution to complete within the time
        allocated by the context's budget (remaining_duration). If the execution does not complete within this time frame, it attempts to cancel
        the future that represents the ongoing execution task.
        
        Args:
            context (ChainContext):
                 The context within which the skill execution takes place. It provides details like
                the time budget and should contain all information required for the skill's execution.
            executor (RunnerExecutor):
                 The executor that handles task submission and provides a way to asynchronously run
                the skill. This parameter expects an object that has a `submit` method compatible with the
                concurrent.futures.Executor interface.
        
        Raises:
            TimeoutError:
                 If the future's result method raises a TimeoutError, indicating that the execution did
                not complete within the remaining_duration specified by the context's budget.
        
        Note:
            The actual implementation of error handling for timeouts or other exceptions during the execution of
            the task should be handled within the `run_in_current_thread` method or inside the executor's
            implementation. This method's focus is on ensuring the task is submitted and respects the execution
            time constraints as specified by the context's budget.
            

        """
        future = executor.submit(self.run_in_current_thread, context, IterationContext.empty())
        try:
            future.result(timeout=context.budget.remaining_duration)
        finally:
            future.cancel()

    def run_in_current_thread(self, context: ChainContext, iteration_context: Option[IterationContext]) -> None:
        """
        Execute the skill within the current thread.
        This method is responsible for executing a skill in the current thread of execution. It sets up a skill context using the provided chain context and iteration context, executes the skill, appends its output to the context, and handles any exceptions that may occur during the execution.
        
        Args:
            context (ChainContext):
                 The chain context containing shared data and state for the current chain execution.
            iteration_context (Option[IterationContext]):
                 The context specific to the current iteration, which might be None or an instance of IterationContext.
        
        Raises:
            RunnerSkillError:
                 If an unexpected error occurs during the execution of the skill, it logs the exception, appends the error message to the context, and rethrows the error wrapped in a RunnerSkillError.
            

        """
        try:
            with SkillContext.from_chain_context(context, iteration_context) as skill_context:
                message = self.execute_skill(skill_context)
                context.append(message)
        except Exception as e:
            context.logger.exception("unexpected error during execution of skill %s", self._name)
            context.append(self.from_exception(e))
            raise RunnerSkillError(f"an unexpected error occurred in skill {self._name}") from e

    @abc.abstractmethod
    def execute_skill(self, context: SkillContext) -> ChatMessage:
        """
        Executes a skill within a given context and returns a chat message as a response.
        This method is an abstract method, meaning it must be implemented by
        subclasses of the class where this method is defined. The `execute_skill` method takes
        a `SkillContext` object, processes it according to the specific skill logic,
        and produces a `ChatMessage` that encapsulates the response.
        
        Args:
            context (SkillContext):
                 An instance of SkillContext that provides relevant
                data and operations for executing the skill. It contains information
                like user input, conversation state, environment settings, and other
                necessary details required to perform the skill.
        
        Returns:
            (ChatMessage):
                 An object representing the message to be delivered back
                to the user. This includes the actual text of the message, any additional
                metadata like attachments or interactive components, and possibly
                instructions for the conversational agent's behavior.

        """
        pass

    def from_exception(self, exception: Exception) -> ChatMessage:
        """
        Generates a ChatMessage from a caught Exception during the execution of a skill.
        This function is used to encapsulate an exception raised by a skill into a ChatMessage object.
        It marks the message as an error and includes the source skill's name in the message content.
        
        Args:
            exception (Exception):
                 The exception that was raised during the skill's execution.
        
        Returns:
            (ChatMessage):
                 A ChatMessage object containing details about the exception, marked as an error,
                with the source attribute set to the name of the skill that raised the exception.

        """
        message = f"skill '{self._name}' raised exception: {exception}"
        return ChatMessage.skill(message, data=None, source=self._name, is_error=True)
