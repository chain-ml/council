from typing import List, Dict, Sequence, TypeVar, Any


class Monitor:
    children: Dict[str, "Monitor"]
    properties: Dict[str, Any]

    def __init__(self, inner: object, **kwargs):
        self.inner = inner
        self.type = inner.__class__.__name__
        self.children = {}
        self.properties = kwargs

    def register_child(self, relation: str, child: "Monitor"):
        self.children[relation] = child


T_co = TypeVar("T_co", bound="Monitorable", covariant=True)


class Monitorable:
    def __init__(self):
        self._monitor = Monitor(self)

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def register_child(self, relation: str, child: "Monitorable"):
        self._monitor.register_child(relation, child._monitor)

    def register_children(self, relation: str, children: Sequence[T_co]) -> None:
        for index, child in enumerate(children):
            self.register_child(f"{relation}[{index}]", child)


def _render_as_text(monitor: Monitor, prefix: str = "", indent: int = 0, indent_step: int = 2) -> List[str]:
    padding = "".join([" " * indent])
    properties = ", ".join([f"{name}: {value}" for name, value in monitor.properties.items()])
    current = f"{padding}{prefix}{monitor.type} ({properties})"
    result = [
        item
        for name, child in monitor.children.items()
        for item in _render_as_text(child, name + ": ", indent + indent_step, indent_step)
    ]

    return [current] + result


def render_as_text(item: Monitorable) -> str:
    return "\n".join(_render_as_text(item.monitor))
