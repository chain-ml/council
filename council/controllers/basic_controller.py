"""

A module encapsulating the implementation of a basic controller for managing execution units within a multi-chain context.

This module provides the `BasicController` class which inherits from `ControllerBase`. The `BasicController` orchestrates the execution of multiple chains of execution units, allowing for parallel or sequential processing based on the controller's configured parallelism state. It offers a simple runtime environment for the execution units by initiating them with a given context and budget. The class ensures that each execution unit within the chains is associated with an appropriate rank which governs the order of their execution.

Classes:
    BasicController(ControllerBase): A controller designed to initiate and manage the execution of a sequence of execution units.



"""
from typing import List, Sequence

from council.contexts import AgentContext
from .controller_base import ControllerBase
from .execution_unit import ExecutionUnit
from ..chains import ChainBase


class BasicController(ControllerBase):
    """
    A controller class that extends ControllerBase to manage the execution of a sequence of ChainBase objects.
    This controller is responsible for orchestrating the execution of multiple chains in a structured manner. It can operate either sequentially or in parallel, depending on the 'parallelism' flag set during initialization.
    
    Attributes:
        chains (Sequence[ChainBase]):
             A sequence of ChainBase instances that the BasicController will manage.
        parallelism (bool):
             A boolean flag indicating whether the chains should be executed in parallel. Defaults to False.
        Constructor:
            Initializes a new instance of the BasicController class.
    
    Args:
        chains (Sequence[ChainBase]):
             A sequence of ChainBase instances that are to be managed by the controller.
        parallelism (bool, optional):
             A boolean flag that indicates whether execution should proceed in parallel. Defaults to False.
    
    Methods:
        _execute(context:
             AgentContext) -> List[ExecutionUnit]: Executes the chains managed by the BasicController.
    
    Args:
        context (AgentContext):
             The context in which the execution should take place, providing necessary information like budget.
    
    Returns:
        (List[ExecutionUnit]):
             A list of ExecutionUnits that represent the execution plan for each chain, including their budgets and ranks.

    """

    def __init__(self, chains: Sequence[ChainBase], parallelism: bool = False):
        """
        Initializes a new instance of the class.
        
        Args:
            chains (Sequence[ChainBase]):
                 A sequence of ChainBase instances that represent the chain of operations or processes to be managed.
            parallelism (bool, optional):
                 A flag indicating whether parallelism should be enabled or not. Defaults to False, meaning parallelism is not enabled by default.

        """
        super().__init__(chains, parallelism)

    def _execute(self, context: AgentContext) -> List[ExecutionUnit]:
        """
        Performs the execution of a list of chains within a given context and returns the resultant execution units each with a specified rank.
        
        Args:
            context (AgentContext):
                 The context of the agent which includes information necessary for execution, such as the budget.
        
        Returns:
            (List[ExecutionUnit]):
                 A list of ExecutionUnit instances, each created for the chains associated with the context.
        
        Raises:
            TypeError:
                 If the context is not of type AgentContext.
            ValueError:
                 If any of the chains within the context cannot be executed due to constraints such as insufficient budget.

        """
        return [ExecutionUnit(chain, context.budget, rank=self.default_execution_unit_rank) for chain in self._chains]
