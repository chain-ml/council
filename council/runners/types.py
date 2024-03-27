"""

Module that defines custom type aliases related to council execution workflows.

This module contains type aliases that are used throughout the council system to define common
data structures and interfaces for executing and handling chain contexts. These types are utilized
to ensure adherence to a specific callable signature and iterability criterion within the system's
execution components.

Attributes:
    RunnerPredicate (Callable[[ChainContext], bool]): Type alias for a function that takes a
        ChainContext as input and returns a boolean. This type is used for functions intended to
        check certain conditions on a ChainContext, typically used to decide if a runner is
        applicable for a given context or not.

    RunnerGenerator (Callable[[ChainContext], Iterable[Any]]): Type alias for a function that
        takes a ChainContext as input and produces an iterable of any type. This type is designed
        for use in generating sequences of items based on the provided ChainContext, which can
        then be consumed by other parts of the council system.


"""
from typing import Callable, Iterable, Any

from council.contexts import ChainContext

RunnerPredicate = Callable[[ChainContext], bool]
RunnerGenerator = Callable[[ChainContext], Iterable[Any]]
