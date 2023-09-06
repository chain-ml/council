import json
from typing import List, Dict, Mapping, Sequence, TypeVar, Any, Generic, Iterable


class Monitor:
    _children: Dict[str, "Monitor"]
    _base_type: str
    _properties: Dict[str, Any]

    def __init__(self, inner: object, base_type: str):
        self.type = inner.__class__.__name__
        self._children = {}
        self._properties = {}
        self._base_type = base_type

    def register_child(self, relation: str, child: "Monitor") -> None:
        self._children[relation] = child

    def set(self, name: str, value: Any) -> None:
        self._properties[name] = value

    def set_name(self, value: str) -> None:
        self._properties["name"] = value

    def set_base_type(self, value: str) -> None:
        self._base_type = value

    @property
    def children(self) -> Mapping[str, "Monitor"]:
        return self._children

    @property
    def properties(self) -> Mapping[str, Any]:
        return self._properties

    @property
    def base_type(self) -> str:
        return self._base_type


T = TypeVar("T", bound="Monitorable")


class Monitored(Generic[T]):
    def __init__(self, name: str, monitorable: T):
        self._name = name
        self._inner = monitorable

    @property
    def inner(self) -> T:
        return self._inner

    @property
    def name(self) -> str:
        return self._name


class Monitorable:
    def __init__(self, base_type: str):
        self._monitor = Monitor(self, base_type)

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def new_monitor(self, name: str, item: T) -> Monitored[T]:
        self.register_child(name, item)
        return Monitored(name, item)

    def new_monitors(self, name: str, items: Iterable[T]) -> List[Monitored[T]]:
        result = [Monitored(f"{name}[{index}]", item) for index, item in enumerate(items)]
        [self.register_child(item.name, item.inner) for item in result]
        return result

    def register_child(self, relation: str, child: "Monitorable"):
        self._monitor.register_child(relation, child._monitor)

    def register_children(self, relation: str, children: Sequence[T]) -> None:
        for index, child in enumerate(children):
            self.register_child(f"{relation}[{index}]", child)


def _render_as_text(monitor: Monitor, prefix: str = "", indent: int = 0, indent_step: int = 2) -> List[str]:
    padding = "".join([" " * indent])
    properties = ", ".join([f"{name}: {value}" for name, value in monitor.properties.items()])
    current = f"{padding}{prefix}{monitor.type}({monitor.base_type}) {{{properties}}}"
    result = [
        item
        for name, child in monitor.children.items()
        for item in _render_as_text(child, name + ": ", indent + indent_step, indent_step)
    ]

    return [current] + result


def _render_as_json(monitor: Monitor) -> Dict[str, Any]:
    children = []
    for name, child in monitor.children.items():
        children.append({"name": name, "value": _render_as_json(child)})
    return {"properties": monitor.properties, "type": monitor.type, "baseType": monitor.base_type, "children": children}


def render_as_text(item: Monitorable) -> str:
    return "\n".join(_render_as_text(item.monitor))


def render_as_json(item: Monitorable) -> str:
    return json.dumps(_render_as_json(item.monitor), indent=2)
