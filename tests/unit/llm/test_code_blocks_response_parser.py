import unittest

from council.llm import LLMParsingException
from council.llm.llm_function import (
    code_blocks_response_parser,
    LLMFunction,
    FunctionOutOfRetryError,
)
from council.mocks import MockLLM, MockMultipleResponses


@code_blocks_response_parser
class Response:
    text: str
    flag: bool
    age: int
    number: float

    def validate(self) -> None:
        if self.age < 0:
            raise LLMParsingException(f"Age must be a positive number; got `{self.age}`")


@code_blocks_response_parser
class BadResponse:
    complex_type: Response


def execute_mock_llm_func(llm, response_parser, max_retries=0):
    llm_func = LLMFunction(llm, response_parser=response_parser, system_message="", max_retries=max_retries)
    return llm_func.execute(user_message="")


class TestCodeBlocksResponseParser(unittest.TestCase):
    def test_no_block(self):
        llm = MockLLM.from_response("")

        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("`text` block is not found")

    def test_wrong_bool(self):
        llm = MockLLM.from_response(
            """
```text
Some text
```

```flag
not-a-bool
```

```age
34
```

```number
3.14
```
"""
        )
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("Cannot convert value `not-a-bool` to bool for field `flag`")

    def test_wrong_int(self):
        llm = MockLLM.from_response(
            """
```text
Some text
```

```flag
true
```

```age
not-an-int
```

```number
3.14
```
"""
        )
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("Cannot convert value `not-an-int` to int for field `age`")

    def test_validate_int(self):
        llm = MockLLM.from_response(
            """
```text
Some text
```

```flag
true
```

```age
-5
```

```number
3.14
```
"""
        )
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("Age must be a positive number; got `-5`")

    def test_wrong_float(self):
        llm = MockLLM.from_response(
            """
```text
Some text
```

```flag
true
```

```age
34
```

```number
not-a-float
```
"""
        )
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, Response.from_response)

        assert str(e.exception).strip().endswith("Cannot convert value `not-a-float` to float for field `number`")

    def test_wrong_type(self):
        llm = MockLLM.from_response(
            """
```complex_type
Some text
```
"""
        )
        with self.assertRaises(FunctionOutOfRetryError) as e:
            _ = execute_mock_llm_func(llm, BadResponse.from_response)

        assert str(e.exception).strip().endswith("Unsupported type `Response` for field `complex_type`")

    def test_correct(self):
        llm = MockLLM.from_response(
            """
```text
Some text
```

```flag
true
```

```age
34
```

```number
3.14
```
"""
        )
        response = execute_mock_llm_func(llm, Response.from_response)

        self.assertIsInstance(response, Response)
        self.assertEqual(response.text, "Some text")
        self.assertTrue(response.flag)
        self.assertEqual(response.age, 34)
        self.assertEqual(response.number, 3.14)

    def test_correction(self):
        responses = [
            # bad response
            [
                """
```text
Some other text
```

```flag
false
```

```age
34
```
"""
            ],
            # good response simulating self-correction
            [
                """
```text
Sorry, forgot the number
```

```flag
false
```

```age
34
```

```number
123
```
"""
            ],
        ]

        llm = MockLLM(action=MockMultipleResponses(responses=responses))
        response = execute_mock_llm_func(llm, Response.from_response, max_retries=1)

        self.assertIsInstance(response, Response)
        self.assertEqual(response.text, "Sorry, forgot the number")
        self.assertFalse(response.flag)
        self.assertEqual(response.age, 34)
        self.assertEqual(response.number, 123.0)
