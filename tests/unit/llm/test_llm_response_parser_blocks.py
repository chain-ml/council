import unittest

from pydantic import field_validator, Field

from council.llm import LLMParsingException
from council.llm.llm_function import (
    LLMFunction,
    FunctionOutOfRetryError,
    LLMResponse,
    CodeBlocksResponseParser,
    StringResponseParser,
    EchoResponseParser,
)
from council.mocks import MockLLM, MockMultipleResponses


class Response(CodeBlocksResponseParser):
    text: str = Field(..., min_length=1)
    flag: bool
    age: int
    number: float

    @field_validator("text")
    @classmethod
    def n(cls, text: str) -> str:
        if text == "incorrect":
            raise ValueError(f"Incorrect `text` value: `{text}`")
        return text

    def validator(self) -> None:
        if self.age < 0:
            raise LLMParsingException(f"Age must be a positive number; got `{self.age}`")


def format_response(text: str, flag: str, age: str, number: str) -> str:
    return f"""
```text
{text}
```

```flag
{flag}
```

```age
{age}
```

```number
{number}
```
"""


def execute_mock_llm_func(llm, response_parser, max_retries=0):
    llm_func = LLMFunction(llm, response_parser=response_parser, system_message="", max_retries=max_retries)
    return llm_func.execute(user_message="")


class TestEchoResponseParser(unittest.TestCase):
    def test(self) -> None:
        llm = MockLLM.from_response("Some LLM response")
        response = execute_mock_llm_func(llm, EchoResponseParser.from_response, max_retries=0)

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.value, "Some LLM response")


class TestStringResponseParser(unittest.TestCase):
    def test(self) -> None:
        llm = MockLLM.from_response("Some LLM response")
        response = execute_mock_llm_func(llm, StringResponseParser.from_response, max_retries=0)

        self.assertIsInstance(response, str)
        self.assertEqual(response, "Some LLM response")


class TestCodeBlocksResponseParser(unittest.TestCase):
    def test_no_block(self):
        llm = MockLLM.from_response("")

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("`text` block is not found")

    def test_wrong_bool(self):
        llm = MockLLM.from_response(format_response(text="Some text", flag="not-a-bool", age="34", number="3.14"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        self.assertIn(
            "Input should be a valid boolean, unable to interpret input "
            "[type=bool_parsing, input_value='not-a-bool', input_type=str]",
            str(e.exception),
        )

    def test_wrong_int(self):
        llm = MockLLM.from_response(format_response(text="Some text", flag="true", age="not-an-int", number="3.14"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        self.assertIn(
            "Input should be a valid integer, unable to parse string as an integer "
            "[type=int_parsing, input_value='not-an-int', input_type=str]",
            str(e.exception),
        )

    def test_wrong_float(self):
        llm = MockLLM.from_response(format_response(text="Some text", flag="true", age="34", number="not-a-float"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        self.assertIn(
            "Input should be a valid number, unable to parse string as a number "
            "[type=float_parsing, input_value='not-a-float', input_type=str]",
            str(e.exception),
        )

    def test_pydentic_validation(self):
        llm = MockLLM.from_response(format_response(text="incorrect", flag="true", age="34", number="3.14"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        self.assertIn(
            "Value error, Incorrect `text` value: `incorrect` "
            "[type=value_error, input_value='incorrect', input_type=str]",
            str(e.exception),
        )

    def test_non_empty_validator(self):
        llm = MockLLM.from_response(format_response(text="   ", flag="true", age="34", number="3.14"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        self.assertIn(
            "String should have at least 1 character [type=string_too_short, input_value='', input_type=str]",
            str(e.exception),
        )

    def test_custom_validation(self):
        llm = MockLLM.from_response(format_response(text="Some text", flag="true", age="-5", number="3.14"))
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("Age must be a positive number; got `-5`")

    def test_correct(self):
        llm = MockLLM.from_response(format_response(text="Some text", flag="true", age="34", number="3.14"))
        response = execute_mock_llm_func(llm, Response.from_response)

        self.assertIsInstance(response, Response)
        self.assertEqual(response.text, "Some text")
        self.assertTrue(response.flag)
        self.assertEqual(response.age, 34)
        self.assertEqual(response.number, 3.14)

    def test_correction(self):
        responses = [
            # bad response
            [format_response(text="Some text", flag="false", age="34", number="A good choice is 123")],
            # good response simulating self-correction
            [format_response(text="Sorry, let me fix", flag="false", age="34", number="123")],
        ]

        llm = MockLLM(action=MockMultipleResponses(responses=responses))
        response = execute_mock_llm_func(llm, Response.from_response, max_retries=1)

        self.assertIsInstance(response, Response)
        self.assertEqual(response.text, "Sorry, let me fix")
        self.assertFalse(response.flag)
        self.assertEqual(response.age, 34)
        self.assertEqual(response.number, 123.0)
