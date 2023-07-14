from typing import Callable, TypeVar, Optional, Generic, Union

from council.utils import Option, read_env_int, read_env_float

T = Union[str, int]
Validator = Callable[[T], bool]


class Parameter:
    def __init__(self, name: str, required: bool, typ: T, default: Optional[T] = None, validator: Validator = lambda x: True):

        self._name = name
        self._required = required
        self._validator = validator
        self._type = typ
        self._value: Option[T] = Option.none()
        if default is not None and validator(default):
            self._value = Option(default)

    def from_env(self, env_var: str):
        if isinstance(self._type, int):
            v = read_env_int(env_var, required=self._required)
            if self._validator(v.unwrap()):
                self._value = v

    @property
    def value(self) -> Option[T]:
        return self._value
