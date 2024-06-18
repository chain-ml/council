from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml
from council.utils import DataObject, DataObjectSpecBase


class LLMPromptTemplate:
    def __init__(self, template: str, model: Optional[str], model_family: Optional[str]) -> None:
        self._template = template
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

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> LLMPromptTemplate:
        template = values.get("template")
        if template is None:
            raise ValueError("`template` must be defined")

        model = values.get("model", None)
        model_family = values.get("model-family", None)
        return LLMPromptTemplate(template, model, model_family)

    @property
    def template(self) -> str:
        return self._template

    def is_compatible(self, model: str) -> bool:
        if self._model is not None and self._model == model:
            return True

        if self._model_family is not None and model.startswith(self._model_family):
            return True
        return False


class LLMPromptConfigSpec(DataObjectSpecBase):
    def __init__(self, system: Sequence[LLMPromptTemplate], user: Optional[Sequence[LLMPromptTemplate]]) -> None:
        self.system_prompts = list(system)
        self.user_prompts = list(user or [])

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LLMPromptConfigSpec:
        system_prompts = values.get("system", [])
        user_prompts = values.get("user")
        if not system_prompts:
            raise ValueError("System prompt(s) must be defined")

        system = [LLMPromptTemplate.from_dict(p) for p in system_prompts]

        user: Optional[List[LLMPromptTemplate]] = None
        if user_prompts is not None:
            user = [LLMPromptTemplate.from_dict(p) for p in user_prompts]
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
    Helper class to instantiate a LLMPrompt from a YAML file
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
        return bool(self.spec.user_prompts)

    def get_system_prompt_template(self, model: str) -> str:
        return self._get_prompt_template(self.spec.system_prompts, model)

    def get_user_prompt_template(self, model: str) -> str:
        if not self.has_user_prompt_template:
            raise ValueError("No user prompt template provided")
        return self._get_prompt_template(self.spec.user_prompts, model)

    @staticmethod
    def _get_prompt_template(prompts: List[LLMPromptTemplate], model: str) -> str:
        """
        Get the first prompt compatible to the given `model` (or `default` prompt).

        Args:
            prompts (List[LLMPromptTemplate]): List of prompts to search from

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
