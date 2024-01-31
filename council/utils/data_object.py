from __future__ import annotations
from typing_extensions import Self

import abc
from typing import Any, Dict, TypeVar, Generic, Type, Optional

import yaml


class DataObjectMetadata:
    def __init__(self, name: str, labels: Dict[str, Any], description: Optional[str] = None):
        self.name = name
        self.description = description
        self.labels = labels

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def get_label_value(self, label: str) -> Optional[Any]:
        if label in self.labels:
            return self.labels[label]
        return None

    def is_matching_labels(self, labels: Dict[str, Any]) -> bool:
        for label in labels:
            value = self.get_label_value(label)
            if value != labels[label]:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "labels": self.labels}

        if self.description is not None:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> DataObjectMetadata:
        return DataObjectMetadata(values["name"], values.get("labels", {}), values.get("description", None))


T = TypeVar("T", bound="DataObjectSpecBase")


class DataObjectSpecBase(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, values: Dict[str, Any]) -> Self:
        pass


class DataObject(Generic[T]):
    def __init__(self, kind: str, version: str, metadata: DataObjectMetadata, spec: T):
        self.kind = kind
        self.version = version
        self.metadata = metadata
        self.spec = spec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def _check_kind(cls, values: Dict[str, Any], expected: str) -> None:
        kind = values.get("kind", None)
        if kind != expected:
            raise ValueError(f"Expected kind: `{expected}`, found `{kind}` instead.")

    @classmethod
    def _from_dict(cls, inner: Type[T], values: Dict[str, Any]) -> Self:
        metadata = DataObjectMetadata.from_dict(values["metadata"])
        spec = inner.from_dict(values["spec"])
        return cls(values["kind"], values["version"], metadata, spec)

    def to_yaml(self, filename: str):
        values = self.to_dict()
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(values, f, default_flow_style=False)
