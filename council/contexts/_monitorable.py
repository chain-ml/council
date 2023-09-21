import json
from typing import Any, Dict, Iterable, List, TypeVar

from ._monitor import Monitor
from ._monitored import Monitored

T = TypeVar("T", bound="Monitorable")
T_monitored = TypeVar("T_monitored", bound="Monitored")


class Monitorable:
    def __init__(self, base_type: str):
        self._monitor = Monitor(self, base_type)

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def new_monitor(self, name: str, item: T) -> Monitored[T]:
        self._register_child(name, item)
        return Monitored(name, item)

    def register_monitor(self, monitored: T_monitored) -> T_monitored:
        self._register_child(monitored.name, monitored.inner)
        return monitored

    def new_monitors(self, name: str, items: Iterable[T]) -> List[Monitored[T]]:
        result = [Monitored(f"{name}[{index}]", item) for index, item in enumerate(items)]
        [self._register_child(item.name, item.inner) for item in result]
        return result

    def _register_child(self, relation: str, child: "Monitorable"):
        self._monitor.register_child(relation, child._monitor)

    def render_as_text(self) -> str:
        return "\n".join(self.monitor.render_as_text())

    def render_as_dict(self) -> Dict[str, Any]:
        """
        returns the graph of operation as a dictionary
        """
        return self.monitor.render_as_dict()

    def render_as_json(self) -> str:
        """
        returns the graph of operation as a JSON string
        """
        return json.dumps(self.monitor.render_as_dict(), indent=2)
