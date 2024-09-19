import json
import unittest
from typing import Dict, List, Any, Literal, Optional

import yaml
from pydantic import BaseModel

from council.llm.llm_function import FunctionOutOfRetryError
from council.llm.llm_response_parser import YAMLBlockResponseParser, JSONBlockResponseParser
from council.mocks import MockLLM
from tests.unit.llm.test_llm_response_parser_blocks import execute_mock_llm_func


class NumberReasoningPair(BaseModel):
    number: float
    reasoning: str


class YAMLComplexResponse(YAMLBlockResponseParser):
    mode: Literal["mode_one", "mode_two"]
    pairs: List[NumberReasoningPair]
    nested_dict: Dict[str, Any]
    value_with_default: int = 48


class JSONComplexResponse(JSONBlockResponseParser):
    mode: Literal["mode_one", "mode_two"]
    pairs: List[NumberReasoningPair]
    nested_dict: Dict[str, Any]
    value_with_default: int = 48


def format_dict(mode: str, pairs: Optional[List[Dict[str, Any]]], value_with_default: Optional[int]) -> Dict[str, Any]:
    data: Dict[str, Any] = {"mode": mode, "nested_dict": {"abc": "xyz", "inner_list": [1, 2, 3]}}
    if pairs is not None:
        data["pairs"] = pairs
    if value_with_default is not None:
        data["value_with_default"] = value_with_default

    return data


def format_response_yaml(
    mode: str, pairs: Optional[List[Dict[str, Any]]] = None, value_with_default: Optional[int] = None
) -> str:
    return f"```yaml\n{yaml.dump(format_dict(mode, pairs, value_with_default))}\n```"


def format_response_json(
    mode: str, pairs: Optional[List[Dict[str, Any]]] = None, value_with_default: Optional[int] = None
) -> str:
    return f"```json\n{json.dumps(format_dict(mode, pairs, value_with_default))}\n```"


class TestYAMLBlockResponseParser(unittest.TestCase):
    def test_no_block(self):
        llm = MockLLM.from_response("")

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        assert str(e.exception).strip().endswith("yaml block is not found")

    def test_incorrect_yaml(self):
        llm = MockLLM.from_response(
            """
```yaml
this_yaml: is
  not: parsable
```
"""
        )

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        assert "Error while parsing yaml:" in str(e.exception).strip()

    def test_invalid_mode(self):
        llm = MockLLM.from_response(
            format_response_yaml(mode="invalid_mode", pairs=[{"number": 1.0, "reasoning": "test"}])
        )

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        self.assertIn("Input should be 'mode_one' or 'mode_two'", str(e.exception))

    def test_missing_required_field(self):
        llm = MockLLM.from_response(format_response_yaml(mode="mode_one"))

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        self.assertIn("Field required", str(e.exception))
        self.assertIn("pairs", str(e.exception))

    def test_invalid_number_in_pair(self):
        llm = MockLLM.from_response(
            format_response_yaml(mode="mode_one", pairs=[{"number": "not_a_number", "reasoning": "test"}])
        )

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        self.assertIn("Input should be a valid number", str(e.exception))
        self.assertIn("not_a_number", str(e.exception))

    def test_correct(self):
        llm = MockLLM.from_response(
            format_response_yaml(
                mode="mode_one",
                pairs=[{"number": 3.14, "reasoning": "some text"}, {"number": 5.4, "reasoning": "more text"}],
                value_with_default=123,
            )
        )
        response = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        self.assertIsInstance(response, YAMLComplexResponse)
        self.assertEqual(response.mode, "mode_one")
        self.assertIsInstance(response.pairs[0], NumberReasoningPair)
        self.assertEqual(
            response.pairs,
            [
                NumberReasoningPair(number=3.14, reasoning="some text"),
                NumberReasoningPair(number=5.4, reasoning="more text"),
            ],
        )
        self.assertEqual(response.nested_dict["abc"], "xyz")
        self.assertEqual(response.nested_dict["inner_list"], [1, 2, 3])
        self.assertEqual(response.value_with_default, 123)

    def test_default_value(self):
        llm = MockLLM.from_response(
            format_response_yaml(
                mode="mode_one",
                pairs=[{"number": 3.14, "reasoning": "some text"}, {"number": 5.4, "reasoning": "more text"}],
            )
        )
        response = execute_mock_llm_func(llm, YAMLComplexResponse.from_response)

        self.assertIsInstance(response, YAMLComplexResponse)
        self.assertEqual(response.value_with_default, 48)


class TestJSONBlockResponseParser(unittest.TestCase):
    def test_no_block(self):
        llm = MockLLM.from_response("")

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, JSONComplexResponse.from_response)

        assert str(e.exception).strip().endswith("json block is not found")

    def test_incorrect_json(self):
        llm = MockLLM.from_response(
            """
```json
this is non parsable
```
"""
        )

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, JSONComplexResponse.from_response)

        assert "Error while parsing json:" in str(e.exception).strip()

    def test_correct(self):
        llm = MockLLM.from_response(
            format_response_json(mode="mode_one", pairs=[{"number": 7.0, "reasoning": "some text"}])
        )
        response = execute_mock_llm_func(llm, JSONComplexResponse.from_response)

        self.assertIsInstance(response, JSONComplexResponse)
        self.assertEqual(response.mode, "mode_one")
        self.assertIsInstance(response.pairs[0], NumberReasoningPair)
        self.assertEqual(
            response.pairs,
            [NumberReasoningPair(number=7.0, reasoning="some text")],
        )
        self.assertEqual(response.value_with_default, 48)
