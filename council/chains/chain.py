from typing import List, Optional

from council.contexts import ChainContext, Monitorable, Monitored
from council.runners import RunnerBase, RunnerExecutor, Sequential


class Chain(Monitorable):
    """
    Represents a chain of skills that can be executed in a specific order.
    """

    _name: str
    _description: str
    _runner: Monitored[RunnerBase]

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
        super().__init__("chain")
        self._name = name
        self._runner = self.new_monitor("runner", Sequential.from_list(*runners))
        self.monitor.name = name
        self._description = description

    @property
    def runner(self) -> RunnerBase:
        """
        the runner of the chain
        """
        return self._runner.inner

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
        return self.description

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

    def _execute(
        self,
        context: ChainContext,
        executor: Optional[RunnerExecutor] = None,
    ) -> None:
        executor = (
            RunnerExecutor(max_workers=10, thread_name_prefix=f"chain_{self.name}") if executor is None else executor
        )

        self._runner.inner.fork_run_merge(self._runner, context, executor)

    def __repr__(self):
        return f"Chain({self.name}, {self.description})"

    def __str__(self):
        return f"Chain {self.name}, description: {self.description}"
