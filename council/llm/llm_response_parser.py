import json
import re
from typing import Any, Callable, Dict, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..utils import CodeParser
from .llm_answer import LLMParsingException
from .llm_middleware import LLMResponse

T_Response = TypeVar("T_Response")
LLMResponseParser = Callable[[LLMResponse], T_Response]

T = TypeVar("T", bound="BaseModelResponseParser")


class BaseModelResponseParser(BaseModel):
    """Base class for parsing LLM responses into structured data models"""

    @classmethod
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


class YAMLBlockResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single YAML code block"""
        llm_response = response.value

        yaml_block = CodeParser.find_first("yaml", llm_response)
        if yaml_block is None:
            raise LLMParsingException("yaml block is not found")

        yaml_content = YAMLResponseParser.parse(yaml_block.code)
        return cls.create_and_validate(**yaml_content)


class YAMLResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw YAML content"""
        llm_response = response.value

        yaml_content = YAMLResponseParser.parse(llm_response)
        return cls.create_and_validate(**yaml_content)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise LLMParsingException(f"Error while parsing yaml: {e}")


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
