"""

A module for managing and monitoring consumption-based budgets in a computational context, with support for time limits and various other resource limits defined using consumption units. It offers classes for handling different kinds of budgets and consumption units, along with utilities for extending, checking, and operating within preset budget limits. It defines custom exceptions for handling budget expiration scenarios and provides utility functions for default budget setup based on environment variables.

Classes:
    BudgetExpiredException: An exception for signaling that a budget's time limit or other resource constraints have been exceeded.

    Consumption: Represents a single resource consumption unit with a value, unit type, and kind.

    Budget: Encapsulates a time-based budget with an optional list of Consumption objects, each representing a specific limit on resource usage.

        Attributes:
            duration (float): The total duration allocated for the budget.
            deadline (float): The computed time at which the budget will expire.

        Methods:
            is_expired: Checks if the budget has expired based on time or resource consumption.
            can_consume: Determines if a certain amount of a resource can be consumed without exceeding the budget.
            add_consumption: Adds a specified amount of resource consumption to the budget.
            add_consumptions: Adds multiple resource consumption amounts to the budget.

    InfiniteBudget: A special case of Budget with a very long duration, effectively representing an unlimited budget.

Functions:
    read_env_int: Utility function to read an integer from the environment, with support for default values and required settings.


"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from council.utils import read_env_int


class BudgetExpiredException(Exception):
    """
    A custom exception that indicates an event where a budget has expired or been exceeded.
    This exception is derived from Python's built-in Exception class. It does not add any additional functionality or attributes over the standard Exception, but serves as a distinct type of error that can be caught and handled separately from other exceptions. This can be useful in financial or budgeting applications where operations are closely tied to budget constraints and need explicit checks for when a budget period has ended or the allocated funds have been used up.
    
    Attributes:
        Inherits all attributes from the base Exception class.

    """
    pass


class Consumption:
    """
    A class that encapsulates the concept of a consumable resource, tracking the amount consumed, its unit of measurement, and the type of consumption.
    
    Attributes:
        _value (float):
             The numerical quantity consumed.
        _unit (str):
             The unit of measurement for the consumed quantity (e.g., liters, kilowatt-hours).
        _kind (str):
             A description of the consumption kind (e.g., 'water', 'electricity').
    
    Methods:
        __init__(self, value:
             float, unit: str, kind: str) -> None: Initializes a new instance of Consumption class.
        value(self) -> float:
             A property that returns the consumed quantity.
        unit(self) -> str:
             A property that returns the unit of measurement.
        kind(self) -> str:
             A property that returns the description of the consumption kind.
        __str__(self) -> str:
             Provides a string representation of the consumption instance.
        add(self, value:
             float) -> Consumption: Creates a new Consumption instance with an increased consumption value.
        subtract(self, value:
             float) -> Consumption: Creates a new Consumption instance with a decreased consumption value.
        add_value(self, value:
             float) -> None: Increases the consumption value in place.
        subtract_value(self, value:
             float) -> None: Decreases the consumption value in place.
        to_dict(self) -> Dict[str, Any]:
             Converts the consumption information into a dictionary format.

    """

    def __init__(self, value: float, unit: str, kind: str) -> None:
        """
        Initializes a new instance of a class with a specified value, unit, and kind.
        
        Args:
            value (float):
                 The numeric value associated with the instance.
            unit (str):
                 The unit of measurement as a string that describes the value.
            kind (str):
                 A string descriptor that categorizes the type of the instance.
        
        Raises:
            TypeError:
                 If any of the arguments is not of the expected type.
        
        Returns:
            (None):
                 This method does not return a value. It's purpose is to initialize the instance variables.

        """
        self._value = value
        self._unit = unit
        self._kind = kind

    @property
    def value(self) -> float:
        """
        Gets the current value of the property.
        
        Returns:
            (float):
                 The current value stored in the '_value' attribute.

        """
        return self._value

    @property
    def unit(self) -> str:
        """
        Property that retrieves the unit attribute.
        
        Returns:
            (str):
                 A string representing the unit associated with an instance.

        """
        return self._unit

    @property
    def kind(self) -> str:
        """
        
        Returns the kind of the object as a string.
            This property method returns the stored '_kind' attribute, which represents the kind of the object.
        
        Returns:
            (str):
                 The kind of the object.

        """
        return self._kind

    def __str__(self):
        """
        
        Returns a formatted string representing the consumption details of the instance.
            This special method is used to create a string representation of the object
            whenever the object is converted to a string (e.g., by the `str()` function). The
            format includes the kind, value, and unit of consumption.
        
        Returns:
            (str):
                 A string representation of the consumption details, which includes
                the kind of consumption, its value, and the unit of measurement.

        """
        return f"{self._kind} consumption: {self._value} {self.unit}"

    def add(self, value: float) -> Consumption:
        """
        Adds the given value to the consumption's value and returns a new Consumption instance.
        This function creates a new `Consumption` object with the current object's unit and kind,
        but with its value increased by the given `value`. It's useful for when you need
        to represent an increase in consumption without changing the original object.
        
        Args:
            value (float):
                 The amount to be added to the current consumption's value.
        
        Returns:
            (Consumption):
                 A new Consumption object with updated value.
            

        """
        return Consumption(self._value + value, self.unit, self._kind)

    def subtract(self, value: float) -> Consumption:
        """
        Subtracts a specified value from the Consumption object's current value and
        creates a new Consumption object with the result.
        This function subtracts the 'value' argument from the '_value' attribute
        of the Consumption instance and returns a new Consumption object with
        the resultant value while keeping the same 'unit' and 'kind' as the original object.
        
        Args:
            value (float):
                 The amount to be subtracted from the Consumption object's value.
        
        Returns:
            (Consumption):
                 A new Consumption object with the value decreased by 'value'.
            

        """
        return Consumption(self._value - value, self.unit, self._kind)

    def add_value(self, value: float) -> None:
        """
        Adds a specified value to the object's '_value' property.
        This method increments the value of the '_value' attribute by the amount provided in the 'value' parameter.
        
        Args:
            value (float):
                 The amount to be added to the '_value' attribute.
        
        Returns:
            None

        """
        self._value += value

    def subtract_value(self, value: float) -> None:
        """
        Subtracts a given value from the object's '_value' attribute.
        This method takes a float value as an input and subtracts it from the
        '_value' attribute of the object. This operation is performed in-place
        and does not return any value.
        
        Args:
            value (float):
                 The value to be subtracted from the object's '_value'.
        
        Returns:
            None
            

        """
        self._value -= value

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation.
        
        Returns:
            (Dict[str, Any]):
                 A dictionary containing keys and values corresponding to the object's attributes.
                The dictionary includes 'kind', 'unit', and 'value' as keys.

        """
        return {"kind": self.kind, "unit": self.unit, "value": self.value}


class Budget:
    """
    A class for tracking and managing budget constraints over a period of time.
    The `Budget` class provides methods to initialize a budget with a specified
    duration and optional consumption limits. It tracks remaining consumption
    limits and time, allowing the user to check if the budget has expired or
    if a specified consumption amount can be allowed within the current limits.
    
    Attributes:
        _duration (float):
             The total duration in seconds for the budget period.
        _deadline (float):
             The monotonic time at which the budget period will end.
        _limits (List[Consumption]):
             (Optional) A list of consumption limit objects.
        _remaining (List[Consumption]):
             (Optional) A list to track the remaining
            consumption limits.
    
    Methods:
        duration:
             Property to get the set duration of the budget.
        deadline:
             Property to get the deadline monotonic time of the budget.
        remaining_duration:
             Property to get the remaining duration in seconds before the budget expires.
        is_expired:
             Check if the budget period has expired or if any limit has been reached.
        can_consume:
             Check if a specified consumption amount can be allowed within the current limits.
        add_consumption:
             Add a new consumption amount to the budget and update the remaining limits.
        _add_consumption:
             Internal method to handle the addition of consumption to the budget.
        add_consumptions:
             Add multiple consumptions to the budget at once and update remaining limits.
        __repr__:
             Return a string representation of the budget object.
        default:
             A static method to create a default `Budget` object using environment variables.
        

    """

    def __init__(self, duration: float, limits: Optional[List[Consumption]] = None) -> None:
        """
        Initializes a new instance of the class with specified duration and optional limits.
        
        Args:
            duration (float):
                 The duration for which this instance is valid. A deadline is computed based on the current monotonic time.
            limits (Optional[List[Consumption]]):
                 An optional list of Consumption objects that define limitations. If provided, each Consumption object will be added to an internal list of limits.
                The constructor initializes the internal duration to the provided duration, sets a deadline by adding the duration to the current monotonic time, initializes an internal list of limits, and also sets a remaining limits list which is a copy of the provided limits if any, otherwise an empty list.

        """
        self._duration = duration
        self._deadline = time.monotonic() + duration
        self._limits = []
        if limits is not None:
            for limit in limits:
                self._limits.append(Consumption(limit.value, limit.unit, limit.kind))

        self._remaining = limits if limits is not None else []

    @property
    def duration(self) -> float:
        """
        Gets the duration property of the instance.
        This method is a property getter that returns the duration of an instance. The duration is expected to be a floating point number representing a period of time in a relevant unit (e.g., seconds, minutes).
        
        Returns:
            (float):
                 The duration value of the instance.

        """
        return self._duration

    @property
    def deadline(self) -> float:
        """
        
        Returns the deadline value for the object.
            This property represents a deadline associated with the object, which is
            expected to be a floating-point value representing a specific cutoff point or time
            limit. The actual meaning of this deadline value can vary depending on the context
            in which the object is used.
        
        Returns:
            (float):
                 The deadline value stored within the object.

        """
        return self._deadline

    @property
    def remaining_duration(self) -> float:
        """
        
        Returns the remaining duration before the deadline is reached.
            This property calculates the difference between the deadline and the
            current time, as measured by a monotonic clock (to prevent issues with clock
            changes). It is used to determine how much time is left before the deadline
            expires.
        
        Returns:
            (float):
                 The remaining time in seconds before the deadline expires.
                If the deadline has already passed, this will return a negative
                value.
            

        """
        return self._deadline - time.monotonic()

    def is_expired(self) -> bool:
        """
        Checks if the current object has expired based on its deadline and remaining limits.
        This method evaluates whether the deadline for the object has passed by comparing the
        '_deadline' attribute with the current time obtained from 'time.monotonic()'. It also checks
        if any of the limits in '_remaining' has a negative value, which would imply that the object
        is past one of its limits.
        
        Returns:
            (bool):
                 True if the object is expired either by deadline or limit, False otherwise.

        """
        if self._deadline < time.monotonic():
            return True

        return any(limit.value < 0.0 for limit in self._remaining)

    def can_consume(self, value: float, unit: str, kind: str) -> bool:
        """
        Checks whether it is possible to consume the specified amount of a given resource unit and kind without exceeding the defined limits.
        Algorithmically, this method iterates over each resource limit in '_remaining'. If a limit with the matching unit and kind is found, it attempts to subtract the 'value' from this limit. If the result is non-negative, consumption is possible without exceeding the limit, hence returns True. If no matching limit is found, or if the subtraction result is negative, it implies that the consumption is not feasible within the remaining limits, so it returns False.
        
        Args:
            value (float):
                 The amount of the resource to consume.
            unit (str):
                 The unit of the resource being consumed.
            kind (str):
                 The kind of resource being consumed.
        
        Returns:
            (bool):
                 True if consumption is possible without exceeding limits, otherwise False.

        """
        for limit in self._remaining:
            if limit.unit == unit and limit.kind == kind:
                c = limit.subtract(value)
                return c.value >= 0.0
        return True

    def add_consumption(self, value: float, unit: str, kind: str):
        """
        Adds a consumption record with the given value, unit, and kind.
        This method wraps the creation of a Consumption object and registers it in the
        system. A Consumption object characterizes the amount of a resource consumed,
        the unit of measure, and the kind of resource.
        
        Args:
            value (float):
                 The amount of the resource that was consumed.
            unit (str):
                 The unit of measurement for the value parameter, such as 'liters' or 'kilowatts'.
            kind (str):
                 The kind of consumption, indicating what type of resource was used, such as 'water', 'electricity', or 'gas'.
        
        Raises:
            TBD:
                 Include here any specific exceptions that this function might throw (e.g., ValueError for negative values).

        """
        self._add_consumption(Consumption(value=value, unit=unit, kind=kind))

    def _add_consumption(self, consumption: Consumption):
        """
        Adds the resource consumption value to the corresponding remaining limit for a unit and kind.
        This method iterates through the current remaining limits of resources and subtracts
        the value of the specified consumption from the limit that matches the same unit and kind.
        
        Args:
            consumption (Consumption):
                 The consumption object, containing the unit, kind, and value,
                which specifies what and how much is being consumed.
        
        Raises:
            ValueError:
                 If there is no matching limit for the given consumption's unit and kind,
                a ValueError is raised indicating that the consumption cannot be subtracted.
            

        """
        for limit in self._remaining:
            if limit.unit == consumption.unit and limit.kind == consumption.kind:
                limit.subtract_value(consumption.value)

    def add_consumptions(self, consumptions: Iterable[Consumption]) -> None:
        """
        Adds multiple consumptions to the current context or object.
        This method takes an iterable of Consumption objects and iterates over it, calling a
        private method to add each individual Consumption to the current context.
        
        Args:
            consumptions (Iterable[Consumption]):
                 An iterable sequence of Consumption objects
                that need to be added.
        
        Returns:
            (None):
                 This method does not return anything.

        """
        for consumption in consumptions:
            self._add_consumption(consumption)

    def __repr__(self):
        """
        
        Returns the official string representation of the Budget object.
            The `__repr__` method returns a string that would be a valid Python expression to recreate
            an object with the same value. In this case, it returns a string that represents how
            a Budget object can be constructed.
        
        Returns:
            (str):
                 A string representation of the Budget object, which can be used to create a new instance
                with the same duration.
            

        """
        return f"Budget({self._duration})"

    @staticmethod
    def default() -> Budget:
        """
        Retrieve the default Budget instance according to the environment variable.
        This static method fetches the 'COUNCIL_DEFAULT_BUDGET' environment variable value,
        converts it to an integer, and initializes a new Budget instance with that duration.
        If the variable is not found or cannot be converted to an integer, it uses a default
        value of 30.
        
        Returns:
            (Budget):
                 A Budget instance with the specified or default duration.
        
        Raises:
            MissingEnvVariableException:
                 If the 'COUNCIL_DEFAULT_BUDGET' environment variable is
                missing and the 'required' parameter is set to True.
            EnvVariableValueException:
                 If the 'COUNCIL_DEFAULT_BUDGET' environment variable value
                cannot be converted to an integer.
            

        """
        duration = read_env_int("COUNCIL_DEFAULT_BUDGET", required=False, default=30)
        return Budget(duration=duration.unwrap())


class InfiniteBudget(Budget):
    """
    Class representing an "InfiniteBudget", which is a subclass of Budget.
    This class simulates a budget that never expires and has an initially high amount set to 10000. It is designed to mimic
    a scenario where budget constraints are not applicable. As a subclass of the Budget class, it inherits all of its
    functionality but overrides the initialization and the check for budget expiration.
    
    Attributes:
        Inherits all attributes from the Budget class.
    
    Methods:
        __init__:
             Constructs an InfiniteBudget instance by initializing the superclass Budget with a high value.
        is_expired:
             Always returns False, indicating the budget does not expire.
        

    """

    def __init__(self):
        """
        Initializes a new instance of the class by setting up a default value for its initial state.
        This constructor calls the superclass initializer with a fixed value of 10000 to set an initial state or configuration
        for the newly created object. The specific purpose of this value is not described, but it typically represents
        a default capacity, limit, or configuration parameter important for the object's operation.
        
        Raises:
            ValueError:
                 If the superclass initialization fails due to invalid parameters or configuration.
            

        """
        super().__init__(10000)

    def is_expired(self) -> bool:
        """
        Determines whether an entity (such as a token or session) is expired or not.
        
        Returns:
            (bool):
                 False if the entity is not expired, True otherwise.

        """
        return False
