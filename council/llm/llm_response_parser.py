import abc
import json
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


YAML_HINTS: Final[
    str
] = """
- Make sure you respect YAML syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.
- Always wrap string values in double quotes (") to ensure proper parsing, except when using the YAML pipe operator (|) for multi-line strings.
"""

YAML_RESPONSE_PARSER_HINTS: Final[str] = "- Provide your response as a parsable YAML." + YAML_HINTS

YAML_BLOCK_RESPONSE_PARSER_HINTS: Final[str] = "- Provide your response in a single yaml code block." + YAML_HINTS

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
                template_parts.append(f"{field_name}: {{{{{description}}}}}")  # field_name: {{value}} when formatted

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
        template_parts = [YAML_BLOCK_RESPONSE_PARSER_HINTS] if include_hints else []
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
        template_parts = [YAML_RESPONSE_PARSER_HINTS] if include_hints else []
        template_parts.append(cls._to_response_template())
        if include_hints:
            template_parts.extend(["", "Only respond with parsable YAML. Do not output anything else."])
        return "\n".join(template_parts)


class JSONBlockResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single JSON code block"""
        llm_response = response.value

        json_block = CodeParser.find_first("json", llm_response)
        if json_block is None:
            raise LLMParsingException("json block is not found")

        json_content = JSONResponseParser.parse(json_block.code)
        return cls.create_and_validate(**json_content)


class JSONResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw JSON content"""
        llm_response = response.value

        json_content = JSONResponseParser.parse(llm_response)
        return cls.create_and_validate(**json_content)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMParsingException(f"Error while parsing json: {e}")
