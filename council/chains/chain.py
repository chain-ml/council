"""

Module `chain` provides a concrete implementation of `ChainBase` from `council.chains.chain_base`.

This module defines the `Chain` class, which is responsible for managing a sequence of `RunnerBase` instances and
executing them in a specified context. The class extends the functionality of `ChainBase` by adding a
monitoring feature for the runners it manages.

Classes:
    Chain: Encapsulates the execution logic of a sequence of `RunnerBase` instances and is responsible for
    their lifecycle and execution within a `ChainContext`. It supports instructions based on its superclass's
    configuration.

Attributes:
    _runner (Monitored[RunnerBase]): A monitored instance of `RunnerBase` which is the execution unit that
    carries out the tasks defined in the chain. This attribute is not exposed publicly, but can be accessed
    via the `runner` property.

Properties:
    runner (RunnerBase): A public getter property providing access to the internal `_runner` attribute.

Methods:
    __init__(name: str, description: str, runners: Sequence[RunnerBase], support_instructions: bool=False):
        Initializes a new `Chain` instance with the specified name, description, and a sequence of runners.
        Optionally specifies if instruction support should be enabled.

    _execute(context: ChainContext, executor: Optional[RunnerExecutor]=None) -> None:
        Executes the chain's internal runner within the provided `ChainContext`. It uses the given `executor`
        to perform the execution. If no `executor` is provided, it defaults to a new `RunnerExecutor` with
        a predefined number of workers and a thread name prefix.


"""
from typing import Optional, Sequence

from council.chains.chain_base import ChainBase
from council.contexts import ChainContext, Monitored
from council.runners import RunnerBase, RunnerExecutor, Sequential


class Chain(ChainBase):
    """
    A class representing a series of executable runners arranged in a sequence, where each runner is monitored and executed in an environment defined by a common context.
    This class extends the functionality of a basic ChainBase by integrating a monitor for the execution of RunnerBase instances. It allows sequential execution of runners, enabling support for fork, run, and merge operations in a managed execution environment.
    
    Attributes:
        _runner (Monitored[RunnerBase]):
             A monitored runner object that ensures all runner instances in the sequence are observed and managed.
    
    Args:
        name (str):
             The name of the chain.
        description (str):
             A brief description of the chain's purpose.
        runners (Sequence[RunnerBase]):
             An ordered iterable of RunnerBase instances to be executed as part of the chain.
        support_instructions (bool, optional):
             A flag to enable support for special instructions within the runners. Defaults to False.
        Properties:
        runner (RunnerBase):
             Returns the inner runner instance from the monitored runner for external interactions.
    
    Methods:
        _execute(context:
             ChainContext, executor: Optional[RunnerExecutor]=None) -> None:
            Initiates the execution of the runners using the specified executor or creates a new one if none is provided. The executor coordinates the tasks over multiple worker threads and processes the runners according to the given execution context.
    
    Args:
        context (ChainContext):
             The context in which the chain and its runners operate.
        executor (Optional[RunnerExecutor]):
             The executor that manages the runner executions. If None, a new executor with predefined configurations is created.

    """

    _runner: Monitored[RunnerBase]

    def __init__(self, name: str, description: str, runners: Sequence[RunnerBase], support_instructions: bool = False):
        """
        Initializes a new instance of the object with specified parameters.
        
        Args:
            name (str):
                 The name of the object being initialized.
            description (str):
                 A brief description of the object.
            runners (Sequence[RunnerBase]):
                 A sequence of RunnerBase instances that this object will manage.
            support_instructions (bool, optional):
                 Flag indicating whether instructions are supported. Defaults to False.
                The constructor also initializes a new monitor with a Sequential runner composed from the provided 'runners' list.

        """
        super().__init__(name, description, support_instructions)
        self._runner = self.new_monitor("runner", Sequential.from_list(runners))

    @property
    def runner(self) -> RunnerBase:
        """
        
        Returns the inner RunnerBase instance associated with this class.
            This property method returns the inner runner object, which is assumed to be
            an instance of RunnerBase or its subclass, encapsulated within the '_runner' attribute of the class.
        
        Returns:
            (RunnerBase):
                 The inner runner object that handles the underlying running mechanism.
        
        Raises:
            AttributeError:
                 If the '_runner' or its 'inner' attribute does not exist or
                is not of type RunnerBase.

        """
        return self._runner.inner

    def _execute(
        self,
        context: ChainContext,
        executor: Optional[RunnerExecutor] = None,
    ) -> None:
        """
        Execute the chain of operations within a given context, optionally using a provided executor.
        This method will create a default `RunnerExecutor` with specified `max_workers` and
        `thread_name_prefix` if an executor is not supplied. It then delegates the execution to the
        `fork_run_merge` method of the internal `_runner` object.
        
        Args:
            context (ChainContext):
                 The context in which the chain should be executed.
            executor (Optional[RunnerExecutor]):
                 An optional executor to run the tasks. If None is provided,
                a default executor with 10 max_workers and a thread name prefix based on the chain name is used.
        
        Returns:
            (None):
                 This method does not return anything as its main purpose is to trigger the execution process.

        """
        executor = (
            RunnerExecutor(max_workers=10, thread_name_prefix=f"chain_{self.name}") if executor is None else executor
        )

        self._runner.inner.fork_run_merge(self._runner, context, executor)
