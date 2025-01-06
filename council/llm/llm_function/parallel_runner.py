from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, List, Sequence

from council.llm.llm_function.llm_function import T_Response

Execute = Callable[..., T_Response]
Reduce = Callable[[Sequence[T_Response]], T_Response]


class ParallelRunner(Generic[T_Response]):
    """
    Executes a function multiple times in parallel and aggregates the results.

    Args:
        execute (Execute): Function to execute in parallel
        n (int): Number of parallel executions to perform
        reduce (Reduce): Function to aggregate sequence of responses into a single response
    """

    def __init__(self, execute: Execute, n: int, reduce: Reduce) -> None:
        self.runner_execute = execute
        self.n = n
        self.reduce = reduce

        self.results: List[List[T_Response]] = []

    @property
    def last_results(self) -> List[T_Response]:
        return self.results[-1]

    def execute(self, *args, **kwargs) -> T_Response:
        """
        Execute the function n times in parallel and aggregate their results.

        Args:
            *args: Positional arguments to pass to the execute function
            **kwargs: Keyword arguments to pass to the execute function

        Returns:
            T_Response: Aggregated result from all parallel executions
        """
        current_results: List[T_Response] = []

        with ThreadPoolExecutor(max_workers=self.n) as executor:
            # Submit all runners to the executor
            future_to_runner = {
                executor.submit(self.runner_execute, *args, **kwargs): self.runner_execute for _ in range(self.n)
            }

            # Collect results as they complete
            for future in as_completed(future_to_runner):
                try:
                    result = future.result()
                    current_results.append(result)
                except Exception as e:
                    # Cancel any pending futures
                    for f in future_to_runner:
                        f.cancel()
                    raise e

            self.results.append(current_results)

        return self.reduce(current_results)
