import unittest
from council.llm.llm_response_parser import YAMLBlockResponseParser, YAMLResponseParser
from pydantic import Field
from typing import Literal, List


class MissingDescriptionField(YAMLBlockResponseParser):
    number: float = Field(..., description="Number from 1 to 10")
    reasoning: str


multiline_description = "Carefully\nreason about the number"


class YAMLBlockResponse(YAMLBlockResponseParser):
    reasoning: str = Field(..., description=multiline_description)
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")


class YAMLResponse(YAMLResponseParser):
    reasoning: str = Field(..., description=multiline_description)
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")


class YAMLBlockResponseReordered(YAMLBlockResponseParser):
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")
    reasoning: str = Field(..., description=multiline_description)


class YAMLResponseReordered(YAMLResponseParser):
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")
    reasoning: str = Field(..., description=multiline_description)


class YAMLBlockResponseReorderedAgain(YAMLBlockResponseParser):
    abc: str = Field(..., description="Not multiline description")
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    reasoning: str = Field(..., description=multiline_description)


class YAMLResponseReorderedAgain(YAMLResponseParser):
    abc: str = Field(..., description="Not multiline description")
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    reasoning: str = Field(..., description=multiline_description)


class ComplexResponse(YAMLBlockResponseParser):
    mode: Literal["mode_one", "mode_two"] = Field(..., description="Mode of operation, one of `mode_one` or `mode_two`")
    pairs: List[YAMLBlockResponse] = Field(..., description="List of number and reasoning pairs")


class TestYAMLBlockResponseParserResponseTemplate(unittest.TestCase):
    def test_missing_description_field(self):
        with self.assertRaises(ValueError) as e:
            MissingDescriptionField.to_response_template()
        self.assertEqual(str(e.exception), "Description is required for field `reasoning` in MissingDescriptionField")

    def test_number_reasoning_pair_template(self):
        template = YAMLBlockResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description
```""",
        )

    def test_number_reasoning_pair_reordered_template(self):
        template = YAMLBlockResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
number: # Number from 1 to 10
abc: # Not multiline description
reasoning: |
  Carefully
  reason about the number
```""",
        )

    def test_number_reasoning_pair_reordered_again_template(self):
        template = YAMLBlockResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
abc: # Not multiline description
number: # Number from 1 to 10
reasoning: |
  Carefully
  reason about the number
```""",
        )

    def test_complex_response_template(self):
        template = ComplexResponse.to_response_template(include_hints=False)
        # nested objects are not supported yet
        self.assertEqual(
            template,
            """```yaml
mode: # Mode of operation, one of `mode_one` or `mode_two`
pairs: # List of number and reasoning pairs
```""",
        )

    def test_with_hints(self):
        template = YAMLBlockResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response in a single yaml code block.
- Make sure you respect YAML syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.
- Always wrap string values in double quotes (") to ensure proper parsing, except when using the YAML pipe operator (|) for multi-line strings.

```yaml
reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description
```""",
        )


class TestYAMLResponseParserResponseTemplate(unittest.TestCase):
    def test_number_reasoning_pair_template(self):
        template = YAMLResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description""",
        )

    def test_number_reasoning_pair_reordered_template(self):
        template = YAMLResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """number: # Number from 1 to 10
abc: # Not multiline description
reasoning: |
  Carefully
  reason about the number""",
        )

    def test_number_reasoning_pair_reordered_again_template(self):
        template = YAMLResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """abc: # Not multiline description
number: # Number from 1 to 10
reasoning: |
  Carefully
  reason about the number""",
        )

    def test_with_hints(self):
        template = YAMLResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response as a parsable YAML.
- Make sure you respect YAML syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.
- Always wrap string values in double quotes (") to ensure proper parsing, except when using the YAML pipe operator (|) for multi-line strings.

reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description

Only respond with parsable YAML. Do not output anything else.""",
        )
