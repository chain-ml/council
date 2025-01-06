from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import yaml
from council.utils import DataObject, DataObjectSpecBase
from typing_extensions import Self


class PromptTemplateCompatibilityError(Exception):
    """
    Exception raised when a prompt template is not valid (cannot be created from dict),
    but could be valid for another template class.
    """

    pass


class PromptTemplateBase(ABC):
    def __init__(self, *, model: Optional[str], model_family: Optional[str]) -> None:
        """Initialize prompt template with at least one of `model` or `model-family`."""
        self._model: Optional[str] = model
        self._model_family: Optional[str] = model_family

        if self._model is None and self._model_family is None:
            raise ValueError("At least one of `model` or `model-family` must be defined")

        if self._model is not None and self._model_family is not None:
            if not self._model.startswith(self._model_family):
                raise ValueError(
                    f"model `{self._model}` and model-family `{self._model_family}` are not compliant."
                    f"Please use separate prompt templates"
                )

    @property
    @abstractmethod
    def template(self) -> str:
        """Return prompt template as a string."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, values: Dict[str, Any]) -> Self:
        pass

    @staticmethod
    def extract_template(values: Dict[str, Any]) -> Any:
        template = values.get("template")
        if template is None:
            raise ValueError("`template` must be defined")

        return template

    def is_compatible(self, model: str) -> bool:
        """Check if the prompt template is compatible with the given model."""
        if self._model is not None and self._model == model:
            return True

        if self._model_family is not None and model.startswith(self._model_family):
            return True
        return False


class StringPromptTemplate(PromptTemplateBase):
    """Prompt template implementation where template is a simple string."""

    def __init__(self, *, template: str, model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._template = template

    @property
    def template(self) -> str:
        """Return prompt template as a string."""
        return self._template

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> StringPromptTemplate:
        template = cls.extract_template(values)
        if not isinstance(template, str):
            raise PromptTemplateCompatibilityError("`template` must be string for StringPromptTemplate")

        model = values.get("model", None)
        model_family = values.get("model-family", None)
        return StringPromptTemplate(template=template, model=model, model_family=model_family)


class XMLPromptSection:
    """Represents a section in an XML-based prompt."""

    def __init__(self, *, name: str, content: str) -> None:
        self.name = name
        self.name_snake_case = name.lower().strip().replace(" ", "_")
        self.content = content.strip()

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> XMLPromptSection:
        name = values.get("name")
        content = values.get("content")
        if name is None or content is None:
            raise ValueError("Both 'name' and 'content' must be defined")
        return XMLPromptSection(name=name, content=content)

    def to_xml(self) -> str:
        """XML representation of the prompt section."""
        return f"<{self.name_snake_case}>\n{self.content}\n</{self.name_snake_case}>"


class XMLPromptTemplate(PromptTemplateBase):
    """Prompt template implementation where template consists of XML sections."""

    def __init__(
        self, *, template: Sequence[XMLPromptSection], model: Optional[str], model_family: Optional[str]
    ) -> None:
        super().__init__(model=model, model_family=model_family)
        self._sections = list(template)

    @property
    def template(self) -> str:
        """Return prompt template as a string, formatting each section to XML."""
        return "\n".join(section.to_xml() for section in self._sections)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> XMLPromptTemplate:
        template = cls.extract_template(values)
        if not isinstance(template, list):
            raise PromptTemplateCompatibilityError("`template` must be a list of sections")

        xml_sections = [XMLPromptSection.from_dict(section) for section in template]

        model = values.get("model", None)
        model_family = values.get("model-family", None)
        return XMLPromptTemplate(template=xml_sections, model=model, model_family=model_family)


class LLMPromptConfigSpec(DataObjectSpecBase):
    def __init__(self, system: Sequence[PromptTemplateBase], user: Optional[Sequence[PromptTemplateBase]]) -> None:
        self.system_prompts = list(system)
        self.user_prompts = list(user or [])

    @staticmethod
    def _determine_template_class(prompt_dict: Dict[str, Any]) -> Type[PromptTemplateBase]:
        template_classes: List[Type[PromptTemplateBase]] = [XMLPromptTemplate, StringPromptTemplate]
        for template_class in template_classes:
            try:
                template_class.from_dict(prompt_dict)
                return template_class
            except PromptTemplateCompatibilityError:
                continue
            except ValueError as e:
                raise e
        raise ValueError(f"Could not determine template class for prompt: {prompt_dict}")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LLMPromptConfigSpec:
        system_prompts = values.get("system", [])
        if not system_prompts:
            raise ValueError("System prompt(s) must be defined")

        # determine template class based on the first system prompt
        template_class = cls._determine_template_class(system_prompts[0])

        # parse all prompts with determined template class
        try:
            system = [template_class.from_dict(p) for p in system_prompts]
            user_prompts = values.get("user", [])
            user = [template_class.from_dict(p) for p in user_prompts] if user_prompts else None
        except (ValueError, PromptTemplateCompatibilityError) as e:
            raise ValueError(f"Failed to parse prompts with template class {template_class.__name__}: {str(e)}")

        return LLMPromptConfigSpec(system, user)

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


class LLMPromptConfigObject(DataObject[LLMPromptConfigSpec]):
    """
    Helper class to instantiate a LLMPrompt object from a YAML file.
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

    @property
    def has_user_prompt_template(self) -> bool:
        """Return True, if user prompt template was specified in yaml file."""
        return bool(self.spec.user_prompts)

    def get_system_prompt_template(self, model: str) -> str:
        """Return system prompt template for a given model."""
        return self._get_prompt_template(self.spec.system_prompts, model)

    def get_user_prompt_template(self, model: str) -> str:
        """
        Return user prompt template for a given model.
        Raises ValueError if no user prompt template was provided.
        """

        if not self.has_user_prompt_template:
            raise ValueError("No user prompt template provided")
        return self._get_prompt_template(self.spec.user_prompts, model)

    @staticmethod
    def _get_prompt_template(prompts: List[PromptTemplateBase], model: str) -> str:
        """
        Get the first prompt compatible to the given `model` (or `default` prompt).

        Args:
            prompts (List[PromptTemplateBase]): List of prompts to search from

        Returns:
            str: prompt template

        Raises:
            ValueError: if both prompt template for a given model and default prompt template are not provided
        """
        try:
            return next(prompt.template for prompt in prompts if prompt.is_compatible(model))
        except StopIteration:
            try:
                return next(prompt.template for prompt in prompts if prompt.is_compatible("default"))
            except StopIteration:
                raise ValueError(f"No prompt template for a given model `{model}` nor a default one")
