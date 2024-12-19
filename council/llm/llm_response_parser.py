from __future__ import annotations

import abc
import json
import os
import re
from typing import Any, Callable, Dict, Final, Type, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..utils import CodeParser
from .llm_answer import LLMParsingException
from .llm_middleware import LLMResponse

T_Response = TypeVar("T_Response")

LLMResponseParser = Callable[[LLMResponse], T_Response]

T = TypeVar("T", bound="BaseModelResponseParser")

RESPONSE_HINTS_FILE_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "data", "response_hints.yaml")


class ResponseHintsHelper:
    def __init__(self, hints: Dict[str, str], prefix: str):
        self.hints_common = hints[f"{prefix}_hints_common"]
        self.parser_hints_start = hints[f"{prefix}_parser_hints_start"]
        self.parser_hints_end = hints[f"{prefix}_parser_hints_end"]
        self.block_parser_hints_start = hints[f"{prefix}_block_parser_hints_start"]

    @classmethod
    def from_yaml(cls, path: str, prefix: str) -> ResponseHintsHelper:
        with open(path, "r", encoding="utf-8") as file:
            hints = yaml.safe_load(file)
        return cls(hints, prefix)

    @property
    def parser(self) -> str:
        return self.parser_hints_start + self.hints_common

    @property
    def block_parser(self) -> str:
        return self.block_parser_hints_start + self.hints_common

    @property
    def parser_end(self) -> str:
        return self.parser_hints_end


yaml_response_hints = ResponseHintsHelper.from_yaml(RESPONSE_HINTS_FILE_PATH, prefix="yaml")
json_response_hints = ResponseHintsHelper.from_yaml(RESPONSE_HINTS_FILE_PATH, prefix="json")


class EchoResponseParser:
    @staticmethod
    def from_response(response: LLMResponse) -> LLMResponse:
        """LLMFunction ResponseParser returning LLMResponse"""
        return response


class StringResponseParser:
    @staticmethod
    def from_response(response: LLMResponse) -> str:
        """LLMFunction ResponseParser for plain text responses"""
        return response.value


class BaseModelResponseParser(BaseModel, abc.ABC):
    """Base class for parsing LLM responses into structured data models"""

    model_config = ConfigDict(frozen=True)  # to preserve field order

    @classmethod
    @abc.abstractmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """
        Parse an LLM response into a structured data model.
        Must be implemented by subclasses to define specific parsing logic.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def to_response_template(cls: Type[T]) -> str:
        """Convert an instance to the response template for the LLM."""
        raise NotImplementedError()

    def validator(self) -> None:
        """
        Implement custom validation logic for the parsed data.
        Can be overridden by subclasses to add specific validation rules.
        Raise LLMParsingException to trigger local correction.
        Alternatively, use pydantic validation.
        """
        pass

    @classmethod
    def create_and_validate(cls: Type[T], **kwargs) -> T:
        instance = cls._try_create(**kwargs)
        instance.validator()
        return instance

    @classmethod
    def _try_create(cls: Type[T], **kwargs) -> T:
        """
        Attempt to create a BaseModel object instance.
        Raises an LLMParsingException if a ValidationError occurs during instantiation.
        """

        try:
            return cls(**kwargs)
        except ValidationError as e:
            # LLM-friendlier version of pydantic error message without "For further information visit..."
            clean_exception_message = re.sub(r"For further information visit.*", "", str(e))
            raise LLMParsingException(clean_exception_message)


class CodeBlocksResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing multiple named code blocks"""
        llm_response = response.value
        parsed_blocks: Dict[str, Any] = {}

        for field_name in cls.model_fields.keys():
            block = CodeParser.find_first(field_name, llm_response)
            if block is None:
                raise LLMParsingException(f"`{field_name}` block is not found")
            parsed_blocks[field_name] = block.code.strip()

        return cls.create_and_validate(**parsed_blocks)

    @classmethod
    def to_response_template(cls: Type[T], include_hints: bool = True) -> str:
        """
        Generate code blocks response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal code blocks formatting hints.
        """

        template_parts = (
            [
                "- Provide your response in a the following code blocks.",
                "- All keys must be present in the response, even when their values are empty.",
                "- For empty values, include empty quotes (" ") rather than leaving them blank.",
                "- Your output outside of code blocks will not be parsed.",
            ]
            if include_hints
            else []
        )

        for field_name, field in cls.model_fields.items():
            description = field.description
            if description is None:
                raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")

            template_parts.extend([f"```{field_name}", description, "```"])

        return "\n".join(template_parts)


T_YAMLResponseParserBase = TypeVar("T_YAMLResponseParserBase", bound="YAMLResponseParserBase")


class YAMLResponseParserBase(BaseModelResponseParser, abc.ABC):
    @classmethod
    def _to_response_template(cls: Type[T]) -> str:
        """Generate a YAML response template based on the model's fields and their descriptions."""
        template_parts = []

        for field_name, field in cls.model_fields.items():
            description = field.description
            if description is None:
                raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")

            is_multiline = "\n" in description

            if field.annotation is str and is_multiline:
                template_parts.append(f"{field_name}: |")
                for line in description.split("\n"):
                    template_parts.append(f"  {line.strip()}")
            else:
                template_parts.append(f"{field_name}: # {description}")

        return "\n".join(template_parts)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise LLMParsingException(f"Error while parsing yaml: {e}")


class YAMLBlockResponseParser(YAMLResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single YAML code block"""
        llm_response = response.value

        yaml_block = CodeParser.find_first("yaml", llm_response)
        if yaml_block is None:
            raise LLMParsingException("yaml block is not found")

        yaml_content = YAMLResponseParserBase.parse(yaml_block.code)
        return cls.create_and_validate(**yaml_content)

    @classmethod
    def to_response_template(cls: Type[T_YAMLResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate YAML block response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal YAML block formatting hints.
        """
        template_parts = [yaml_response_hints.block_parser] if include_hints else []
        template_parts.extend(["```yaml", cls._to_response_template(), "```"])
        return "\n".join(template_parts)


class YAMLResponseParser(YAMLResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw YAML content"""
        llm_response = response.value

        yaml_content = YAMLResponseParserBase.parse(llm_response)
        return cls.create_and_validate(**yaml_content)

    @classmethod
    def to_response_template(cls: Type[T_YAMLResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate YAML response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal YAML formatting hints.
        """
        template_parts = [yaml_response_hints.parser] if include_hints else []
        template_parts.append(cls._to_response_template())
        if include_hints:
            template_parts.extend(["", yaml_response_hints.parser_end])
        return "\n".join(template_parts)


T_JSONResponseParserBase = TypeVar("T_JSONResponseParserBase", bound="JSONResponseParserBase")


class JSONResponseParserBase(BaseModelResponseParser, abc.ABC):
    @classmethod
    def _to_response_template(cls: Type[T]) -> str:
        """Generate a JSON response template based on the model's fields and their descriptions."""
        template_dict = {}

        for field_name, field in cls.model_fields.items():
            description = field.description
            if description is None:
                raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")

            is_multiline = "\n" in description

            if field.annotation is str and is_multiline:
                template_dict[field_name] = "\n".join(line.strip() for line in description.split("\n"))
            else:
                template_dict[field_name] = description

        return json.dumps(template_dict, indent=2)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMParsingException(f"Error while parsing json: {e}")


class JSONBlockResponseParser(JSONResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single JSON code block"""
        llm_response = response.value

        json_block = CodeParser.find_first("json", llm_response)
        if json_block is None:
            raise LLMParsingException("json block is not found")

        json_content = JSONResponseParserBase.parse(json_block.code)
        return cls.create_and_validate(**json_content)

    @classmethod
    def to_response_template(cls: Type[T_JSONResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate JSON block response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal JSON block formatting hints.
        """
        template_parts = [json_response_hints.block_parser] if include_hints else []
        template_parts.extend(["```json", cls._to_response_template(), "```"])
        return "\n".join(template_parts)


class JSONResponseParser(JSONResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw JSON content"""
        llm_response = response.value

        json_content = JSONResponseParserBase.parse(llm_response)
        return cls.create_and_validate(**json_content)

    @classmethod
    def to_response_template(cls: Type[T_JSONResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate JSON response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal JSON formatting hints.
        """
        template_parts = [json_response_hints.parser] if include_hints else []
        template_parts.append(cls._to_response_template())
        if include_hints:
            template_parts.extend(["", json_response_hints.parser_end])
        return "\n".join(template_parts)
