from typing import Callable, Iterable, Any

from council.contexts import ChainContext
from .budget import Budget

RunnerPredicate = Callable[[ChainContext, Budget], bool]
RunnerGenerator = Callable[[ChainContext, Budget], Iterable[Any]]
