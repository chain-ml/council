"""

Module for initializing the Chain package.

This module imports and provides easier access to the core classes used for representing
and executing chains in a monitored environment. The chain structures defined herein
serve as base components for more complex logic or workflows.

Classes:
    ChainBase (Monitorable, abc.ABC): An abstract base class that defines the structure
        and mandatory methods for chain-like objects. Implementations of this class must
        provide the _execute method.
    Chain (ChainBase): A concrete implementation of ChainBase that manages a sequence
        of RunnerBase objects. It provides facilities to execute each runner in the chain
        with proper context and executor handling.



"""
from .chain_base import ChainBase
from .chain import Chain
