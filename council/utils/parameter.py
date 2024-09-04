from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

from council.utils import Option, read_env_float, read_env_int, read_env_str

T = TypeVar("T")
Validator = Callable[[T], None]
_Converter = Callable[[str, bool], T]


def greater_than_validator(value: int) -> Validator:
    def validator(x: int) -> None:
        if x <= value:
            raise ValueError(f"must be greater than {value}")

    return validator


def prefix_validator(value: str) -> Validator:
    def validator(x: str) -> None:
        if not x.startswith(value):
            raise ValueError(f"must start with `{value}`")

    return validator


def prefix_any_validator(values: Iterable[str]) -> Validator:
    def validator(x: str) -> None:
        if not any(x.startswith(value) for value in values):
            raise ValueError(f"must start with one of: `{', '.join(v for v in values)}`")

    return validator


def not_empty_validator(x: str) -> None:
    if len(x.strip()) == 0:
        raise ValueError("must not be empty")


class Undefined:
    """
    A class used to distinguish between an undefined value and a defined Optional value set to None
    """

    def __repr__(self) -> str:
        return "Undefined"


OptionalOrUndefined = Union[Optional[T], Undefined]

_undefined = Undefined()


class ParameterValueException(Exception):
    """
    Custom exception raised when a required environment variable is missing.
    """

    def __init__(self, name: str, value: Any, message: Exception) -> None:
        """
        Initializes an instance of ParameterValueException.

        Parameters:
            name (str): The name of the parameter.
            value (Any): The invalid value of the parameter.
            message (Exception): The exception raised

        Returns:
            None
        """
        super().__init__(f"'{name}' parameter value '{value}' is invalid. Value must be {message}")


class Parameter(Generic[T]):
    def __init__(
        self,
        name: str,
        required: bool,
        converter: _Converter,
        value: OptionalOrUndefined[T] = _undefined,
        default: OptionalOrUndefined[T] = _undefined,
        validator: Optional[Validator] = None,
    ) -> None:
        self._name = name
        self._required = required
        self._validator: Validator = validator if validator is not None else lambda x: None
        self._default = default
        if isinstance(value, Undefined):
            if not isinstance(default, Undefined):
                self.set(default)
            else:
                self._value: Option[T] = Option.none()
        else:
            self.set(value)

        self._read_env = converter

    def from_env(self, env_var: str) -> None:
        v = self._read_env(env_var, self._required)
        if v.is_some():
            self.set(v.unwrap())

    def set(self, value: Optional[T]) -> None:
        try:
            self._validator(value)
            self._value = Option(value)
        except ValueError as e:
            raise ParameterValueException(self._name, value=value, message=e)

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Optional[T]:
        return self._value.unwrap() if self.is_some() else None

    @property
    def required(self) -> bool:
        return self._required

    @property
    def is_default(self) -> bool:
        if isinstance(self._default, Undefined):
            return False
        if self._value.is_none():
            return False
        return self._value.unwrap() == self._default

    def unwrap(self) -> T:
        return self._value.unwrap()

    def unwrap_or(self, value: Any) -> Union[T, Any]:
        return self._value.unwrap_or(value)

    def is_some(self) -> bool:
        return self._value.is_some()

    def is_none(self) -> bool:
        return self._value.is_none()

    def __str__(self) -> str:
        opt = "(optional)" if not self._required else ""
        val = "undefined value" if self.is_none() else f"value `{self._value.unwrap()}`"
        default = f" Default value `{self._default}`." if not isinstance(self._default, Undefined) else ""
        return f"Parameter{opt} `{self._name}` with {val}.{default}"

    def __eq__(self, other: Any) -> bool:
        if self.is_none():
            if isinstance(other, Parameter):
                return other.is_none()
            return False

        if isinstance(other, Parameter):
            return self.unwrap() == other.unwrap()
        return self.unwrap() == other

    @staticmethod
    def string(
        name: str,
        required: bool,
        value: OptionalOrUndefined[str] = _undefined,
        default: OptionalOrUndefined[str] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[str]:
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_str,
            default=default,
            validator=validator,
        )

    @staticmethod
    def int(
        name: str,
        required: bool,
        value: OptionalOrUndefined[int] = _undefined,
        default: OptionalOrUndefined[int] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[int]:
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_int,
            default=default,
            validator=validator,
        )

    @staticmethod
    def float(
        name: str,
        required: bool,
        value: OptionalOrUndefined[float] = _undefined,
        default: OptionalOrUndefined[float] = _undefined,
        validator: Optional[Validator] = None,
    ) -> Parameter[float]:
        return Parameter(
            name=name,
            required=required,
            value=value,
            converter=read_env_float,
            default=default,
            validator=validator,
        )
