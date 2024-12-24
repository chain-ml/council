from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import yaml
from council.utils import DataObject, DataObjectSpecBase


class XMLSection:
    def __init__(self, *, name: str, content: str) -> None:
        self.name = name
        self.name_snake_case = name.lower().strip().replace(" ", "_")
        self.content = content.strip()

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> XMLSection:
        name = values.get("name")
        content = values.get("content")
        if name is None or content is None:
            raise ValueError("Both 'name' and 'content' must be defined")
        return XMLSection(name=name, content=content)

    def to_xml(self) -> str:
        return f"<{self.name_snake_case}>\n{self.content}\n</{self.name_snake_case}>"


class XMLPromptSpec(DataObjectSpecBase):
    def __init__(self, system: Sequence[XMLSection], user: Optional[Sequence[XMLSection]] = None) -> None:
        if not system:
            raise ValueError("At least one system section must be defined")

        self.system_sections = list(system)
        self.user_sections = list(user) if user else []

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> XMLPromptSpec:
        system_sections = values.get("system", [])
        if not system_sections:
            raise ValueError("System section(s) must be defined")

        user_sections = values.get("user", [])

        system = [XMLSection.from_dict(s) for s in system_sections]
        user = [XMLSection.from_dict(s) for s in user_sections] if user_sections else None

        return XMLPromptSpec(system, user)

    def to_dict(self) -> Dict[str, Any]:
        result = {"system": self.system_sections}
        if self.user_sections:
            result["user"] = self.user_sections
        return result

    def __str__(self):
        msg = f"{len(self.system_sections)} system section(s)"
        if self.user_sections is not None:
            msg += f"; {len(self.user_sections)} user section(s)"
        return msg


class XMLPromptFormatter(DataObject[XMLPromptSpec]):
    """
    Helper class to format sections into XML format from a YAML file
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> XMLPromptFormatter:
        return super()._from_dict(XMLPromptSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> XMLPromptFormatter:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "XMLPrompt")
            return XMLPromptFormatter.from_dict(values)

    @property
    def has_user_prompt_template(self) -> bool:
        """Return True, if user prompt template was specified in yaml file."""
        return bool(self.spec.user_sections)

    def get_system_prompt_template(self) -> str:
        """Return system prompt template formatted as XML."""
        return self._format_sections(self.spec.system_sections)

    def get_user_prompt_template(self) -> str:
        """
        Return user prompt template for a given model.
        Raises ValueError if no user prompt template was provided.
        """

        if not self.has_user_prompt_template:
            raise ValueError("No user prompt template provided")
        return self._format_sections(self.spec.user_sections)

    @staticmethod
    def _format_sections(sections: Sequence[XMLSection]) -> str:
        return "\n".join(section.to_xml() for section in sections)
