import ast
from typing import List

from council.contexts import ChatMessage, SkillContext
from council.skills import SkillBase

from .llm_helper import extract_code_block

NO_EDIT_BEFORE_THIS_LINE = "# COUNCIL NO EDIT BEFORE THIS LINE"
NO_EDIT_AFTER_THIS_LINE = "# COUNCIL NO EDIT AFTER THIS LINE"

ERROR_CODE_STARTS_WITH = """Code must starts with:
```python
{expected}
```
"""

ERROR_CODE_ENDS_WITH = """Code must ends with:
```python
{expected}
```
"""


class PythonCodeVerificationSkill(SkillBase):
    """
    Skill that verifies a given python code. It verifies:
    - the code is parseable using `ast.parse`
    - the code follows an optional code template.

    The python code is retrieved from the message content from `context.try_last_message`,
    looking for a markdown `python` code block.

    Below is an example of code template::

        def say_hi() -> str:
        # COUNCIL NO EDIT BEFORE THIS LINE

            pass

        # COUNCIL NO EDIT AFTER THIS LINE

        print(say_hi())

    The two `magic` comments `# COUNCIL ...` are used to identify the reference code,
    respectively above and below each `magic` comment.

    The verification process will fail if any non-empty lines of the reference code are missing in the given code.

    The verification process relies on `ast.parse` and `ast.unparse` to standardize the code.
    As such, the reference code must be formatted in such a way it is not impacted by the standardization process,
    a.k.a the reference code must be stable.

    Below is a non-exhaustive list for good practices for the reference code:

    - no comments, other than the `magic` comments
    - empty lines are ok as they will be ignored

    """

    def __init__(self, code_template: str = "") -> None:
        """
        initialize a new instance

        Args:
            code_template: a code template use to validate the python code retrieved from the context.
        """

        super().__init__("code_verification")
        self._code_template = "\n".join(self.normalize_snippet(code_template))

        match = code_template.find(NO_EDIT_BEFORE_THIS_LINE)
        if match >= 0:
            snippet = self.normalize_snippet(code_template[:match])
            self._code_before_line_count = len(snippet)
            self._code_before = "\n".join(snippet)
        else:
            self._code_before_line_count = 0
            self._code_before = ""

        match = code_template.rfind(NO_EDIT_AFTER_THIS_LINE)
        if match >= 0:
            snippet = self.normalize_snippet(code_template[match + len(NO_EDIT_AFTER_THIS_LINE) :])
            self._code_after_line_count = len(snippet)
            self._code_after = "\n".join(snippet)
        else:
            self._code_after_line_count = 0
            self._code_after = ""

        try:
            self._validate_code(self._code_template)
        except Exception as e:
            raise Exception("code template is not stable") from e

    @staticmethod
    def normalize_snippet(snippet: str) -> List[str]:
        lines = snippet.splitlines()
        lines = [line.rstrip() for line in lines]
        return [line for line in lines if line != ""]

    def execute(self, context: SkillContext) -> ChatMessage:
        last_message = context.try_last_message.unwrap("last message")

        try:
            python_code = extract_code_block(last_message.message, "python")
            if python_code == self._code_template:
                return self.build_error_message("generated code cannot be identical to the code template")
            normalized_code = self.normalize_code(python_code)
            self._validate_code(normalized_code)
            return self.build_success_message(f"```python\n{python_code}\n```\n")
        except Exception as e:
            error = f"{e.__class__.__name__}: {e}"
            context.logger.debug(error)
            return self.build_error_message(error)

    def _validate_code(self, code: str) -> None:
        errors = []
        code_lines = self.normalize_snippet(code)
        if self._code_before_line_count > 0:
            actual = "\n".join(code_lines[: self._code_before_line_count])
            if not actual == self._code_before:
                errors.append(ERROR_CODE_STARTS_WITH.format(expected=self._code_before, actual=actual))

        if self._code_after_line_count > 0:
            actual = "\n".join(code_lines[-self._code_after_line_count :])
            if not actual == self._code_after:
                errors.append(ERROR_CODE_ENDS_WITH.format(expected=self._code_after, actual=actual))

        if len(errors) > 0:
            raise Exception("\n".join(errors))

    @staticmethod
    def normalize_code(code: str) -> str:
        module = ast.parse(code, type_comments=True)
        return ast.unparse(module).strip() + "\n"
