from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional

import yaml
from council.utils import CodeParser


class LLMParsingException(Exception):
    def __init__(self, message: str = "Your response is not correctly formatted.") -> None:
        super().__init__(message)
        self.message = message


class llm_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None) -> None:
        super().__init__(fget, fset, fdel, doc)
        self.rank = inspect.getsourcelines(fget)[1]


class llm_class_validator:
    def __init__(self, func: Callable) -> None:
        self.f = func


class LLMProperty:
    def __init__(self, name: str, prop: llm_property) -> None:
        self._name = name
        self._type = prop.fget.__annotations__.get("return", str)
        self._description = prop.__doc__
        self._rank = prop.rank

    @property
    def name(self) -> str:
        return self._name

    @property
    def rank(self) -> int:
        return self._rank

    def __str__(self) -> str:
        return f"{self._name}: {{{self._description}, type: `{self._type.__name__}`}}"

    def can_parse(self, value: Any) -> bool:
        try:
            _ = self._type(value)
            return True
        except (TypeError, ValueError):
            return False

    def parse(self, value: Any, default: Optional[Any]) -> Any:
        def converter(x: str) -> bool:
            result = x.strip().lower()
            if result in ["true", "1", "t"]:
                return True
            if result in ["false", "0", "f"]:
                return False
            raise TypeError(x)

        try:
            if self._type is bool:
                return converter(value)
            return self._type(value)
        except (TypeError, ValueError) as e:
            if default is not None:
                return default
            raise LLMParsingException(f"Value {value} cannot be parsed into {self._type.__name__}") from e


class LLMAnswer:
    def __init__(self, schema: Any) -> None:
        self._schema = schema
        self._class_name = schema.__name__
        self._valid_func = None
        properties = []
        getmembers = inspect.getmembers(schema)
        for attr_name, attr_value in getmembers:
            if isinstance(attr_value, llm_property):
                prop_info = LLMProperty(name=attr_name, prop=attr_value)
                properties.append(prop_info)
            if isinstance(attr_value, llm_class_validator):
                self._valid_func = attr_value.f
        properties.sort(key=lambda item: item.rank)
        self._properties = properties

    @staticmethod
    def field_separator() -> str:
        return "<->"

    def to_prompt(self) -> str:
        p = [f"{prop}" for prop in self._properties]
        return self.field_separator().join(p)

    def to_yaml_prompt(self) -> str:
        fp = [
            "Use precisely the following template:",
            "```yaml",
            f"your yaml formatted answer for the `{self._class_name}` class.",
            "```",
            "\n",
        ]
        p = [f"  {prop}" for prop in self._properties]
        return "\n".join(fp) + self._class_name + ":\n" + "\n".join(p) + "\n"

    def to_object(self, line: str) -> Optional[Any]:
        d = self.parse_line(line, None)
        missing_keys = [key.name for key in self._properties if key.name not in d.keys()]
        if len(missing_keys) > 0:
            raise LLMParsingException(f"Missing `{missing_keys}` in response.")
        t = self._schema(**d)
        if self._valid_func is not None:
            self._valid_func(t)
        return t

    def parse_line(self, line: str, default: Optional[Any] = "Invalid") -> Dict[str, Any]:
        property_value_pairs = line.split(self.field_separator())
        properties_dict = {}
        for pair in property_value_pairs:
            if ":" not in pair:
                continue
            values = pair.split(":", 1)
            prop_name = values[0].replace("'", "")
            prop_name = prop_name.replace("-", "")
            prop_name = prop_name.strip()
            prop_value = values[1].strip()

            class_prop = self._find(prop_name)
            if class_prop is not None:
                typed_value = class_prop.parse(prop_value, default)
                properties_dict[class_prop.name] = typed_value
        return properties_dict

    def parse_yaml(self, block: str) -> Dict[str, Any]:
        d = yaml.safe_load(block)
        properties_dict = {**d}
        missing_keys = [key.name for key in self._properties if key.name not in properties_dict.keys()]
        if len(missing_keys) > 0:
            raise LLMParsingException(f"Missing `{missing_keys}` in response.")
        return properties_dict

    def parse_yaml_list(self, block: str) -> List[Dict[str, Any]]:
        result = []
        d = yaml.safe_load(block)
        for item in d:
            properties_dict = {**item}
            missing_keys = [key.name for key in self._properties if key.name not in properties_dict.keys()]
            if len(missing_keys) > 0:
                raise LLMParsingException(f"Missing `{missing_keys}` in response.")
            result.append(properties_dict)
        return result

    def parse_yaml_bloc(self, block: str) -> Dict[str, Any]:
        code_block = CodeParser.find_first(language="yaml", text=block)
        if code_block is not None:
            return self.parse_yaml(code_block.code)
        return {}

    def _find(self, prop: str) -> Optional[LLMProperty]:
        for p in self._properties:
            if p.name.casefold() == prop.casefold():
                return p
        return None
