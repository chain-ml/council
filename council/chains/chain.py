from typing import List, Any, Optional

from council.contexts import ChainContext
from council.runners import Budget, RunnerBase, Sequential, RunnerExecutor


class Chain:
    """
    Represents a chain of skills that can be executed in a specific order.

    Attributes:
        name (str): The name of the chain.
        description (str): The description of the chain.
        runner (RunnerBase): The runner responsible for executing the chain.
    """

    name: str
    description: str
    runner: RunnerBase

    def __init__(self, name: str, description: str, runners: List[RunnerBase]):
        """
        Initializes the Chain object.

        Args:
            name (str): The name of the chain.
            description (str): The description of the chain.
            runners (List[RunnerBase]): The list of runners representing the skills in the chain.

        Raises:
            None
        """
        self.name = name
        self.runner = Sequential.from_list(*runners)
        self.description = description

    def get_description(self) -> str:
        """
        Retrieves the description of the chain.

        Returns:
            str: The description of the chain.

        Raises:
            None
        """
        return self.description

    def execute(
        self,
        context: ChainContext,
        budget: Budget,
        executor: Optional[RunnerExecutor] = None,
    ) -> Any:
        """
        Executes the chain of skills based on the provided context, budget, and optional executor.

        Args:
            context (ChainContext): The context for executing the chain.
            budget (Budget): The budget for chain execution.
            executor (Optional[RunnerExecutor]): The skill executor to use for executing the chain.

        Returns:
            Any: The result of executing the chain.

        Raises:
            None
        """
        executor = (
            RunnerExecutor(max_workers=10, thread_name_prefix=f"chain_{self.name}") if executor is None else executor
        )
        self.runner.run_from_chain_context(context, budget, executor)

    def __repr__(self):
        return f"Chain({self.name}, {self.description})"

    def __str__(self):
        return f"Chain {self.name}, description: {self.description}"
