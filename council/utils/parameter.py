from typing import Callable, TypeVar, Optional, Generic, Any, Union

from council.utils import Option, read_env_int, read_env_float, read_env_str

T = TypeVar("T")
Validator = Callable[[T], None]
_Converter = Callable[[str, bool], T]


class ParameterValueException(Exception):
    """
    Custom exception raised when a required environment variable is missing.
    """

    def __init__(self, name: str, value: Any, message: Exception):
        """
        Initializes an instance of MissingEnvVariableError.

        Parameters:
            name (str): The name of the missing environment variable.

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
        default: Optional[T] = None,
        validator: Optional[Validator] = None,
    ):
        self._name = name
        self._required = required
        self._value: Option[T] = Option.none()
        self._read_env = converter
        self._validator: Validator = validator if validator is not None else lambda x: None
        if default is not None:
            self.set(default)

    def from_env(self, env_var: str):
        v = self._read_env(env_var, self._required)
        if v.is_some():
            self.set(v.unwrap())

    def set(self, value: T):
        try:
            self._validator(value)
            self._value = Option(value)
        except ValueError as e:
            raise ParameterValueException(self._name, value=value, message=e)

    @property
    def name(self):
        return self._name

    @property
    def value(self) -> Optional[T]:
        return self._value.unwrap() if self.is_some() else None

    @property
    def required(self) -> bool:
        return self._required

    def unwrap(self) -> T:
        return self._value.unwrap()

    def unwrap_or(self, value: Any) -> Union[T, Any]:
        return self._value.unwrap_or(value)

    def is_some(self) -> bool:
        return self._value.is_some()

    def is_none(self) -> bool:
        return self._value.is_none()

    @staticmethod
    def string(
        name: str, required: bool, default: Optional[str] = None, validator: Optional[Validator] = None
    ) -> "Parameter[str]":
        return Parameter(name, required, read_env_str, default, validator)

    @staticmethod
    def int(
        name: str, required: bool, default: Optional[int] = None, validator: Optional[Validator] = None
    ) -> "Parameter[int]":
        return Parameter(name, required, read_env_int, default, validator)

    @staticmethod
    def float(
        name: str, required: bool, default: Optional[float] = None, validator: Optional[Validator] = None
    ) -> "Parameter[float]":
        return Parameter(name, required, read_env_float, default, validator)
