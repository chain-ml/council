from typing import Callable, Iterable, Any

from council.contexts import ChainContext

RunnerPredicate = Callable[[ChainContext], bool]
RunnerGenerator = Callable[[ChainContext], Iterable[Any]]
