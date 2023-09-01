from typing import Callable, Iterable, Any

from council.contexts import Budget, ChainContext

RunnerPredicate = Callable[[ChainContext, Budget], bool]
RunnerGenerator = Callable[[ChainContext, Budget], Iterable[Any]]
