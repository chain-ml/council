from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
from council.utils import DataObject, DataObjectSpecBase
from typing_extensions import Self


class PromptTemplateBase(ABC):
    """Base class for all prompt types"""

    def __init__(self, *, model: Optional[str], model_family: Optional[str]) -> None:
        """Initialize prompt template with at least one of `model` or `model-family`."""
        self._model = model
        self._model_family = model_family

        if self._model is None and self._model_family is None:
            raise ValueError("At least one of `model` or `model-family` must be defined")

        if self._model is not None and self._model_family is not None:
            if not self._model.startswith(self._model_family):
                raise ValueError(
                    f"model `{self._model}` and model-family `{self._model_family}` are not compliant."
                    f"Please use separate prompt templates"
                )

    def is_compatible(self, model: str) -> bool:
        """Check if the prompt template is compatible with the given model."""
        if self._model is not None and self._model == model:
            return True

        if self._model_family is not None and model.startswith(self._model_family):
            return True
        return False

    @property
    @abstractmethod
    def template(self) -> str:
        """Return prompt template as a string."""
        pass


class LLMPromptTemplate(PromptTemplateBase):
    def __init__(self, *, template: str, model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._template = template

    @property
    def template(self) -> str:
        return self._template

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMPromptTemplate:
        template = values.get("template")
        if template is None:
            raise ValueError("`template` must be defined")

        model = values.get("model")
        model_family = values.get("model-family")
        return cls(template=template, model=model, model_family=model_family)


class PromptSection:
    """
    Represents a section in a section-based prompt, e.g. XML, markdown, etc.
    Consists of a name, optional content, and optional nested sections.
    """

    def __init__(
        self, *, name: str, content: Optional[str] = None, sections: Optional[Iterable[PromptSection]] = None
    ) -> None:
        self.name = name
        self.content = content.strip() if content else None
        self.sections = list(sections) if sections else []

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> PromptSection:
        name = values.get("name")
        if name is None:
            raise ValueError("`name` must be defined")

        content = values.get("content")
        sections = [PromptSection.from_dict(section) for section in values.get("sections", [])]

        return PromptSection(name=name, content=content, sections=sections)


class PromptFormatter(ABC):
    """Base formatter interface"""

    def format(self, sections: List[PromptSection]) -> str:
        return "\n".join(self._format_section(section) for section in sections)

    @abstractmethod
    def _format_section(self, section: PromptSection) -> str:
        pass


class StringPromptFormatter(PromptFormatter):
    def __init__(self, section_prefix: str = ""):
        self.section_prefix = section_prefix

    def _format_section(self, section: PromptSection) -> str:
        parts = [f"{self.section_prefix}{section.name}"]
        if section.content:
            parts.append(section.content)
        parts.extend([self._format_section(sec) for sec in section.sections])
        return "\n".join(parts)


class MarkdownPromptFormatter(PromptFormatter):
    def _format_section(self, section: PromptSection, indent: int = 1) -> str:
        parts = [f"{'#' * indent} {section.name}", ""]
        if section.content:
            parts.extend([section.content, ""])
        parts.extend([self._format_section(sec, indent + 1) for sec in section.sections])
        return "\n".join(parts)


class XMLPromptFormatter(PromptFormatter):
    def _format_section(self, section: PromptSection, indent: str = "") -> str:
        indent_diff = "  "
        name_snake_case = section.name.lower().replace(" ", "_")
        parts = [f"{indent}<{name_snake_case}>"]

        if section.content:
            content_lines = section.content.split("\n")
            content = "\n".join([f"{indent}{indent_diff}{line}" for line in content_lines])
            parts.append(content)

        parts.extend([self._format_section(sec, indent + indent_diff) for sec in section.sections])
        parts.append(f"{indent}</{name_snake_case}>")
        return "\n".join(parts)


class LLMStructuredPromptTemplate(PromptTemplateBase):
    def __init__(self, sections: Iterable[PromptSection], *, model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._sections = list(sections)

        self._formatter: PromptFormatter = StringPromptFormatter()

    def set_formatter(self, formatter: PromptFormatter) -> None:
        self._formatter = formatter

    @property
    def template(self) -> str:
        return self._formatter.format(self._sections)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMStructuredPromptTemplate:
        sections = values.get("sections", [])
        if not sections:
            raise ValueError("`sections` must be defined")

        sections = [PromptSection.from_dict(sec) for sec in sections]

        model = values.get("model")
        model_family = values.get("model-family")
        return cls(sections=sections, model=model, model_family=model_family)


class LLMPromptConfigSpecBase(DataObjectSpecBase):
    def __init__(self, system: Sequence[PromptTemplateBase], user: Optional[Sequence[PromptTemplateBase]]) -> None:
        self.system_prompts = list(system)
        self.user_prompts = list(user or [])

    def to_dict(self) -> Dict[str, Any]:
        result = {"system": self.system_prompts}
        if not self.user_prompts:
            result["user"] = self.user_prompts
        return result

    def __str__(self):
        msg = f"{len(self.system_prompts)} system prompt(s)"
        if self.user_prompts is not None:
            msg += f"; {len(self.user_prompts)} user prompt(s)"
        return msg

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LLMPromptConfigSpecBase:
        system_prompts = values.get("system", [])
        user_prompts = values.get("user")
        if not system_prompts:
            raise ValueError("System prompt(s) must be defined")

        system = [cls._prompt_template_from_dict(prompt) for prompt in system_prompts]

        user: Optional[List[PromptTemplateBase]] = None
        if user_prompts is not None:
            user = [cls._prompt_template_from_dict(prompt) for prompt in user_prompts]
        return cls(system, user)

    @staticmethod
    def _prompt_template_from_dict(prompt_dict: Dict[str, Any]) -> PromptTemplateBase:
        raise NotImplementedError("Subclasses must implement this method")


class LLMPromptConfigSpec(LLMPromptConfigSpecBase):
    @staticmethod
    def _prompt_template_from_dict(prompt_dict: Dict[str, Any]) -> PromptTemplateBase:
        return LLMPromptTemplate.from_dict(prompt_dict)


class LLMStructuredPromptConfigSpec(LLMPromptConfigSpecBase):
    @staticmethod
    def _prompt_template_from_dict(prompt_dict: Dict[str, Any]) -> PromptTemplateBase:
        return LLMStructuredPromptTemplate.from_dict(prompt_dict)


class LLMPromptConfigObjectBase(DataObject[LLMPromptConfigSpecBase]):
    @classmethod
    def from_yaml(cls, filename: str) -> Self:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def has_user_prompt_template(self) -> bool:
        """Return True, if user prompt template was specified in yaml file."""
        return bool(self.spec.user_prompts)

    def get_system_prompt_template(self, model: str = "default") -> str:
        """Return system prompt template for a given model."""
        return self._get_prompt_template(self.spec.system_prompts, model)

    def get_user_prompt_template(self, model: str = "default") -> str:
        """
        Return user prompt template for a given model.
        Raises ValueError if no user prompt template was provided.
        """

        if not self.has_user_prompt_template:
            raise ValueError("No user prompt template provided")
        return self._get_prompt_template(self.spec.user_prompts, model)

    @staticmethod
    def _get_prompt_template(prompts: Sequence[PromptTemplateBase], model: str) -> str:
        """
        Get the first prompt compatible to the given `model` (or `default` prompt).

        Args:
            prompts (List[PromptTemplateBase]): List of prompts to search from

        Returns:
            str: prompt template

        Raises:
            ValueError: if both prompt template for a given model and default prompt template are not provided
        """

        compatible_prompt = next((prompt for prompt in prompts if prompt.is_compatible(model)), None)
        if compatible_prompt:
            return compatible_prompt.template

        default_prompt = next((prompt for prompt in prompts if prompt.is_compatible("default")), None)
        if default_prompt:
            return default_prompt.template

        raise ValueError(f"No prompt template for a given model `{model}` nor a default one")


class LLMPromptConfigObject(LLMPromptConfigObjectBase):
    """
    Helper class to instantiate a LLMPrompt from a YAML file.
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMPromptConfigObject:
        return super()._from_dict(LLMPromptConfigSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMPromptConfigObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMPrompt")
            return LLMPromptConfigObject.from_dict(values)


class LLMStructuredPromptConfigObject(LLMPromptConfigObjectBase):
    """
    Helper class to instantiate a LLMStructuredPrompt from a YAML file.
    """

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMStructuredPromptConfigObject:
        return super()._from_dict(LLMStructuredPromptConfigSpec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMStructuredPromptConfigObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
            cls._check_kind(values, "LLMStructuredPrompt")
            return LLMStructuredPromptConfigObject.from_dict(values)

    def set_formatter(self, formatter: PromptFormatter) -> None:
        for prompts in [self.spec.system_prompts, self.spec.user_prompts]:
            for prompt in prompts:
                if isinstance(prompt, LLMStructuredPromptTemplate):
                    prompt.set_formatter(formatter)
