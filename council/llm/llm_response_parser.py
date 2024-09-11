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
    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """Implement parsing functionality"""
        raise NotImplementedError()

    def validator(self) -> None:
        """Implement custom validation functionality - raise LLMParsingException to trigger local correction"""
        pass

    @classmethod
    def create_and_validate(cls: Type[T], **kwargs) -> T:
        instance = cls._try_create(**kwargs)
        instance.validator()
        return instance

    @classmethod
    def _try_create(cls: Type[T], **kwargs) -> T:
        """Try to create BaseModel object instance and raise LLMParsingException if any ValidationError occurs"""

        try:
            return cls(**kwargs)
        except ValidationError as e:
            # LLM-friendlier version of pydantic error message without "For further information visit..."
            clean_exception_message = re.sub(r"For further information visit.*", "", str(e))
            raise LLMParsingException(clean_exception_message)


class CodeBlocksResponseParser(BaseModelResponseParser):
    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
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
        llm_response = response.value

        yaml_block = CodeParser.find_first("yaml", llm_response)
        if yaml_block is None:
            raise LLMParsingException("yaml block is not found")

        yaml_content = YAMLResponseParser.parse(yaml_block.code)
        return cls.create_and_validate(**yaml_content)


class YAMLResponseParser(BaseModelResponseParser):
    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
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
        llm_response = response.value

        json_block = CodeParser.find_first("json", llm_response)
        if json_block is None:
            raise LLMParsingException("json block is not found")

        json_content = JSONResponseParser.parse(json_block.code)
        return cls.create_and_validate(**json_content)


class JSONResponseParser(BaseModelResponseParser):
    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        llm_response = response.value

        json_content = JSONResponseParser.parse(llm_response)
        return cls.create_and_validate(**json_content)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMParsingException(f"Error while parsing json: {e}")
