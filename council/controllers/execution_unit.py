"""

Module `execution_unit` provides a class `ExecutionUnit` representing an execution unit with chain, budget, and other related attributes.

Classes:
    ExecutionUnit: A construct to encapsulate the operational parameters of an execution unit within a system, typically associated with request processing or task execution.

Attributes:
    - chain (ChainBase): The chain associated with the execution unit.
    - budget (Budget): The budget designated for the execution unit's operation.
    - initial_state (Optional[ChatMessage]): The initial state or context that the execution unit begins with. This can be `None`.
    - name (str): A human-readable identifier for the execution unit. Defaults to the chain's name if not provided.
    - rank (int): A rank signifying the execution unit's precedence or importance, defaults to -1 if not provided.

Properties:
    - chain: Returns the associated `ChainBase` instance.
    - budget: Returns the associated `Budget` instance.
    - initial_state: Returns the `ChatMessage` instance representing the initial state, or `None`.
    - name: Returns the execution unit's name as a string.
    - rank: Returns the execution unit's rank as an integer.

The `ExecutionUnit` class manages the assignments and interactions of the various components it encapsulates, ensuring that they work together to achieve their designated tasks within the council's framework. It may keep track of execution-related metrics, handle context transitions, and manage resources according to the budget constraints.


"""
from __future__ import annotations

from typing import Optional

from council.chains import ChainBase
from council.contexts import Budget, ChatMessage


class ExecutionUnit:
    """
    A class that represents an execution unit, which is essentially a component responsible for executing part of a process chain.
    The ExecutionUnit object encapsulates a chain, budget, and optionally a starting state, a name, and a rank. The chain dictates
    the sequence of operations to perform, while the budget is a constraint indicating the resources available for execution (e.g., time, cost).
    The initial state can be supplied as a ChatMessage object to set the starting point of the execution, and a name can be assigned to the
    unit for identification. If no name is provided, the name of the chain is used by default. The rank can be used to order execution units,
    and it defaults to -1 if not specified.
    
    Attributes:
        _chain (ChainBase):
             A reference to the chain of operations to be executed.
        _budget (Budget):
             The resource constraints defined for executing the chain.
        _initial_state (Optional[ChatMessage]):
             The initial state from which the execution will commence if provided.
        _name (str):
             The name assigned to the execution unit for identification. Defaults to the chain's name.
        _rank (int):
             An ordering rank for the execution unit if required. Defaults to -1.
    
    Methods:
        chain (property):
             Returns the associated chain of operations.
        budget (property):
             Returns the budget constraint for the execution.
        initial_state (property):
             Returns the initial state if it was provided during initialization.
        name (property):
             Returns the name of the execution unit.
        rank (property):
             Returns the rank order of the execution unit.

    """

    def __init__(
        self,
        chain: ChainBase,
        budget: Budget,
        initial_state: Optional[ChatMessage] = None,
        name: Optional[str] = None,
        rank: Optional[int] = None,
    ):
        """
        Initializes a new instance of the class with the provided chain, budget, and optional parameters.
        
        Args:
            chain (ChainBase):
                 The chain to which the instance will be associated.
            budget (Budget):
                 The budget allocated for the instance's operations.
            initial_state (Optional[ChatMessage]):
                 The initial state of the chat message, defaults to None.
                This parameter allows for setting an initial state if available.
            name (Optional[str]):
                 An optional human-readable name for the instance. If not provided, the name
                defaults to the name of the chain.
            rank (Optional[int]):
                 An optional ranking value for the instance. If not provided, it defaults to -1,
                indicating that no specific rank has been assigned.
        
        Attributes:
            _chain (ChainBase):
                 Internal storage for the associated chain.
            _budget (Budget):
                 Internal storage for the allocated budget.
            _initial_state (Optional[ChatMessage]):
                 Internal storage for the initial chat message state.
            _name (str):
                 Internal storage for the instance's name.
            _rank (int):
                 Internal storage for the instance's rank.
            

        """
        self._chain = chain
        self._budget = budget
        self._initial_state = initial_state
        self._name = name or chain.name
        self._rank = rank or -1

    @property
    def chain(self) -> ChainBase:
        """
        
        Returns the blockchain instance that the object is associated with.
            The 'chain' property is designed to provide access to the blockchain instance, often referred to as 'ChainBase',
            that the current object is linked with. This could represent the underlying structure or database that
            keeps the blockchain's data consistent and allows for operations such as querying blocks, transactions,
            and various state-related information.
        
        Returns:
            (ChainBase):
                 The blockchain instance associated with the current object.
        
        Raises:
            AttributeError:
                 If the '_chain' attribute is not present in the object, accessing this property
                may raise an AttributeError.

        """
        return self._chain

    @property
    def budget(self) -> Budget:
        """
        Gets the budget associated with the instance.
        This property method returns the current '_budget' attribute of the instance, which
        is expected to be of type 'Budget'.
        
        Returns:
            (Budget):
                 The current budget of the instance.

        """
        return self._budget

    @property
    def initial_state(self) -> Optional[ChatMessage]:
        """
        Gets the initial state of a chat message if it exists.
        This property method returns an `Optional[ChatMessage]`, which can be either the initial state
        of the chat message or `None` if the initial state is not set or does not exist.
        
        Returns:
            (Optional[ChatMessage]):
                 The initial chat message state or `None`.

        """
        return self._initial_state

    @property
    def name(self) -> str:
        """
        Property that gets the current value of the private name attribute.
        This property is used to access the value of the `_name` attribute in an encapsulated way.
        Accessing this property returns the value of the `_name` attribute without directly exposing
        the attribute itself. This is typically used to implement read-only access to a value.
        
        Returns:
            (str):
                 The current value of the `_name` attribute.

        """
        return self._name

    @property
    def rank(self) -> int:
        """
        Gets the rank of an object.
        This is a property that should be used to retrieve the rank value of an instance. The rank is expected to be an integer that represents some sort of level, position, or priority.
        
        Returns:
            (int):
                 An integer representing the rank of the instance. The specific meaning of 'rank' can vary depending on context within the program.

        """
        return self._rank
