import abc
from typing import Optional

from council.contexts import ChainContext, Monitorable
from council.runners import RunnerExecutor


class ChainBase(Monitorable, abc.ABC):
    """
    base class for implementing a Chain
    """

    def __init__(self, name: str, description: str, support_instructions: bool = False) -> None:
        super().__init__("chain")
        self._name: str = name
        self._description: str = description
        self._instructions: bool = support_instructions
        self.monitor.name = name

    @property
    def name(self) -> str:
        """
        the name of the chain
        """
        return self._name

    @property
    def description(self) -> str:
        """
        the description of the chain.
        """
        return self._description

    @property
    def is_supporting_instructions(self) -> bool:
        return self._instructions

    def execute(self, context: ChainContext, executor: Optional[RunnerExecutor] = None) -> None:
        """
        Executes the chain of skills based on the provided context, budget, and optional executor.

        Args:
            context (ChainContext): The context for executing the chain.
            executor (Optional[RunnerExecutor]): The skill executor to use for executing the chain.

        Returns:
            Any: The result of executing the chain.

        Raises:
            None
        """
        with context:
            self._execute(context, executor)

    @abc.abstractmethod
    def _execute(self, context: ChainContext, executor: Optional[RunnerExecutor] = None) -> None:
        pass

    def __repr__(self) -> str:
        return f"Chain({self.name}, {self.description})"

    def __str__(self) -> str:
        return f"Chain {self.name}, description: {self.description}"
