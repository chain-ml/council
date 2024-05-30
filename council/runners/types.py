from typing import Any, Callable, Iterable

from council.contexts import ChainContext

RunnerPredicate = Callable[[ChainContext], bool]
RunnerGenerator = Callable[[ChainContext], Iterable[Any]]
