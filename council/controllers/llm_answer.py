import inspect
import re
from typing import Any, Dict, Optional

import yaml


class llm_property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget, fset, fdel, doc)
        self.rank = inspect.getsourcelines(fget)[1]


class LLMProperty:
    def __init__(self, name: str, prop: llm_property):
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

    def __str__(self):
        return f"{self._name}: {{{self._description}, expected response type `{self._type.__name__}`}}"

    def can_parse(self, value: Any) -> bool:
        try:
            _ = self._type(value)
            return True
        except (TypeError, ValueError):
            return False

    def parse(self, value: Any, default: Optional[Any]) -> Any:
        try:
            return self._type(value)
        except (TypeError, ValueError):
            if default is not None:
                return default
            raise


class LLMControllerAnswer:
    def __init__(self, schema: Any):
        self._schema = schema
        self._class_name = schema.__name__
        properties = []
        getmembers = inspect.getmembers(schema)
        for attr_name, attr_value in getmembers:
            if isinstance(attr_value, llm_property):
                prop_info = LLMProperty(name=attr_name, prop=attr_value)
                properties.append(prop_info)
        properties.sort(key=lambda item: item.rank)
        self._properties = properties

    @staticmethod
    def field_separator() -> str:
        return "<->"

    def to_prompt(self) -> str:
        fp = [
            "\n# FORMATTING",
            # f"For each {self._class_name} follow exactly the following format:\n",
        ]
        p = [f"{prop}" for prop in self._properties]
        return "\n".join(fp) + "`" + self.field_separator().join(p) + "`"

    def to_yaml_prompt(self) -> str:
        fp = [
            "\n# FORMATTING",
            "Use precisely the following template:",
            "```yaml",
            "{your yaml formatted answer}",
            "```",
            "\n",
        ]
        p = [f"  {prop}" for prop in self._properties]
        return "\n".join(fp) + self._class_name + ":\n" + "\n".join(p) + "\n"

    def to_object(self, line: str) -> Optional[Any]:
        d = self.parse_line(line, None)
        missing_keys = [key.name for key in self._properties if key.name not in d.keys()]
        if len(missing_keys) > 0:
            raise Exception(f"missing {missing_keys} in response.")
        t = self._schema(**d)
        return t

    def parse_line(self, line: str, default: Optional[Any] = "Invalid") -> Dict[str, Any]:
        property_value_pairs = line.split(self.field_separator())
        properties_dict = {}
        for pair in property_value_pairs:
            prop_name, value = pair.split(":")
            value = value.strip()
            prop_name = prop_name.replace("'", "")
            class_prop = self._find(prop_name)
            if class_prop is not None:
                typed_value = class_prop.parse(value, default)
                properties_dict[class_prop.name] = typed_value
        return properties_dict

    def parse_yaml(self, bloc: str) -> Dict[str, Any]:
        d = yaml.safe_load(bloc)
        properties_dict = {**d}
        return properties_dict

    def _find(self, prop: str) -> Optional[LLMProperty]:
        for p in self._properties:
            if p.name.casefold() == prop.casefold():
                return p
        return None

    @staticmethod
    def try_extract_bloc(text: str, bloc_type: str) -> Optional[str]:
        bloc = None
        pattern = rf"```{bloc_type}\s+(.*?)\s+```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                bloc = match
        return bloc