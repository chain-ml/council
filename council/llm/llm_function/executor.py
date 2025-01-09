from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, List, Sequence

from .llm_function import T_Response

Execute = Callable[..., T_Response]
Reduce = Callable[[Sequence[T_Response]], T_Response]


class ParallelExecutor(Generic[T_Response]):
    """
    Executes a function (e.g. LLMFunction.execute) multiple times in parallel
    and aggregates the results according to a reduce function.

    Args:
        execute (Execute): Function to execute in parallel
        reduce (Reduce): Function to aggregate sequence of responses into a single response
        n (int): Number of parallel executions to perform
    """

    def __init__(self, execute: Execute, reduce: Reduce, n: int) -> None:
        self.execute_func = execute
        self.n = n
        self.reduce = reduce

    def execute_all(self, *args, **kwargs) -> List[T_Response]:
        """
        Execute the function n times in parallel and return all results.

        Args:
            *args: Positional arguments to pass to the execute function
            **kwargs: Keyword arguments to pass to the execute function

        Returns:
            List[T_Response]: List of all parallel execution results
        """
        results: List[T_Response] = []

        with ThreadPoolExecutor(max_workers=self.n) as executor:
            # Submit all runners to the executor
            future_to_runner = {
                executor.submit(self.execute_func, *args, **kwargs): self.execute_func for _ in range(self.n)
            }

            # Collect results as they complete
            for future in as_completed(future_to_runner):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Cancel any pending futures
                    for f in future_to_runner:
                        f.cancel()
                    raise e

        return results

    def execute(self, *args, **kwargs) -> T_Response:
        """
        Execute the function n times in parallel and aggregate their results.

        Args:
            *args: Positional arguments to pass to the execute function
            **kwargs: Keyword arguments to pass to the execute function

        Returns:
            T_Response: Aggregated result from all parallel executions
        """
        results = self.execute_all(*args, **kwargs)
        return self.reduce(results)
