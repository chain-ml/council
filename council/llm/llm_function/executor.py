from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generic, List, Optional, Sequence, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")

ExecuteFn = Callable[..., T]
ReduceFn = Callable[[Sequence[T]], R]


class ParallelExecutor(Generic[T, R]):
    """
    Executes one or more functions (e.g. LLMFunction.execute) in parallel
    and aggregates the results according to a reduce function.

    Args:
        execute_fns (Union[ExecuteFn, Sequence[ExecuteFn]]):
            Single function or sequence of functions to execute in parallel.
        reduce_fn (Optional[ReduceFn]):
            Function to aggregate sequence of responses into a response of potentially different type.
            Defaults to returning the first result.
        n (int): Number of times to execute if single function provided. Ignored if sequence provided.

    Raises:
        ValueError: If _execute_fns sequence is empty or contains non-callable items.

    .. mermaid::

        flowchart LR
            E1(ExecuteFn 1) --> R(ReduceFn)
            E3(ExecuteFn ...) --> R
            E4(ExecuteFn N) --> R
    """

    def __init__(
        self, execute_fns: Union[ExecuteFn, Sequence[ExecuteFn]], reduce_fn: Optional[ReduceFn] = None, n: int = 1
    ) -> None:
        self._execute_fns: List[ExecuteFn] = self._validate_execute_fns(execute_fns, n)
        self._reduce_fn: ReduceFn = reduce_fn if reduce_fn is not None else lambda results: results[0]

    @staticmethod
    def _validate_execute_fns(execute_fns: Union[ExecuteFn, Sequence[ExecuteFn]], n: int) -> List[ExecuteFn]:
        if not isinstance(execute_fns, Sequence):
            if not callable(execute_fns):
                raise ValueError("`executes` must be callable or sequence of callables")
            return [execute_fns] * n
        else:
            if not execute_fns:
                raise ValueError("`executes` sequence cannot be empty")
            if not all(callable(ex) for ex in execute_fns):
                raise ValueError("All items in `executes` must be callable")
            return list(execute_fns)

    def execute(self, *args, **kwargs) -> Sequence[T]:
        """
        Execute functions in parallel and return all results.

        Args:
            *args: Positional arguments to pass to the execute functions
            **kwargs: Keyword arguments to pass to the execute functions

        Returns:
            Sequence[T]: All execution results
        """
        results: List[T] = []

        with ThreadPoolExecutor(max_workers=len(self._execute_fns)) as executor:
            future_to_runner = {executor.submit(execute, *args, **kwargs): execute for execute in self._execute_fns}

            for future in as_completed(future_to_runner):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    for f in future_to_runner:
                        f.cancel()
                    raise e

        return results

    def reduce(self, results: Sequence[T]) -> R:
        return self._reduce_fn(results)

    def execute_and_reduce(self, *args, **kwargs) -> R:
        """
        Execute the function(s) in parallel and aggregate their results.

        Args:
            *args: Positional arguments to pass to the execute functions
            **kwargs: Keyword arguments to pass to the execute functions

        Returns:
            R: Result from reducing all parallel executions, potentially of different type than inputs
        """
        results = self.execute(*args, **kwargs)
        return self.reduce(results)
