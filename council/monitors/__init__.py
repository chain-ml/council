from datetime import datetime
from typing import List, Dict, Sequence, TypeVar, Any, Generic


class Monitor:
    children: Dict[str, "Monitor"]
    properties: Dict[str, Any]

    def __init__(self, inner: object, **kwargs):
        self.type = inner.__class__.__name__
        self.children = {}
        self.properties = kwargs

    def register_child(self, relation: str, child: "Monitor"):
        self.children[relation] = child


T_co = TypeVar("T_co", bound="Monitorable", covariant=True)


class Monitored(Generic[T_co]):
    def __init__(self, name: str, monitorable: T_co):
        self._name = name
        self._inner = monitorable

    @property
    def inner(self) -> T_co:
        return self._inner

    @property
    def name(self) -> str:
        return self._name


class Monitorable:
    def __init__(self):
        self._monitor = Monitor(self)

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def new_monitor(self, name: str, item: "Monitorable") -> Monitored[T_co]:
        self.register_child(name, item)
        return Monitored(name, item)

    def register_child(self, relation: str, child: "Monitorable"):
        self._monitor.register_child(relation, child._monitor)

    def register_children(self, relation: str, children: Sequence[T_co]) -> None:
        for index, child in enumerate(children):
            self.register_child(f"{relation}[{index}]", child)


class ExecutionLogEntry:
    def __init__(self, source: str, method: str = ""):
        self._source = source
        self._method = method
        self._start = datetime.utcnow()
        self._duration = 0
        self._error = None
        self._consumptions = []

    @property
    def source(self) -> str:
        return self._source

    def log_consumption(self, consumption: Any):
        self._consumptions.append(consumption)

    def log_consumptions(self, consumptions: Sequence[Any]):
        [self.log_consumption(item) for item in consumptions]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._duration = (datetime.utcnow() - self._start).total_seconds()
        self._error = exc_val

    def __repr__(self):
        return f"ExecutionLogEntry(source={self._source}, method={self._method}, start={self._start}, duration={self._duration}, error={self._error})"


class ExecutionLog:
    def __init__(self):
        self._entries = []

    def new_entry(self, name: str) -> ExecutionLogEntry:
        result = ExecutionLogEntry(name)
        self._entries.append(result)
        return result


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
