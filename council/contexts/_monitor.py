from __future__ import annotations

from typing import Any, Dict, List, Mapping


class Monitor:

    def __init__(self, inner: object, base_type: str) -> None:
        self._type = inner.__class__.__name__
        self._children: Dict[str, Monitor] = {}
        self._properties: Dict[str, Any] = {}
        self._base_type: str = base_type

    def register_child(self, relation: str, child: Monitor) -> None:
        self._children[relation] = child

    def set(self, name: str, value: Any) -> None:
        self._properties[name] = value

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._properties["name"]

    @name.setter
    def name(self, value: str) -> None:
        self._properties["name"] = value

    @property
    def children(self) -> Mapping[str, Monitor]:
        return self._children

    @property
    def properties(self) -> Mapping[str, Any]:
        return self._properties

    @property
    def base_type(self) -> str:
        return self._base_type

    def render_as_text(self, prefix: str = "", indent: int = 0, indent_step: int = 2) -> List[str]:
        padding = " " * indent
        properties = ", ".join([f"{name}: {value}" for name, value in self.properties.items()])
        current = f"{padding}{prefix}{self.type}({self.base_type}) {{{properties}}}"
        result = [
            item
            for name, child in self.children.items()
            for item in child.render_as_text(name + ": ", indent + indent_step, indent_step)
        ]

        return [current] + result

    def render_as_dict(self, include_children: bool = True) -> Dict[str, Any]:
        children = []
        if include_children:
            for name, child in self.children.items():
                children.append({"name": name, "value": child.render_as_dict()})
        return {"properties": self.properties, "type": self.type, "baseType": self.base_type, "children": children}
