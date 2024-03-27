"""

Python module for verifying code snippets against a predefined template within a chat message context.

This module includes a class `PythonCodeVerificationSkill` which extends `SkillBase`. It employs
the `extract_code_block` function from the `llm_helper` module to extract code blocks from chat messages.
The skill checks if the code conforms to a template which enforces certain parts of the code to be
unchanged (using `NO_EDIT_BEFORE_THIS_LINE` and `NO_EDIT_AFTER_THIS_LINE` markers).

During initialization, the class stores the normalized version of the code template and its segmented parts.
It validates the template to ensure stability. The `execute` method processes an incoming chat message,
attempting to extract a Python code block and then validating the code against the stored template, while
also normalizing it using the provided `normalize_code` static method.

Exceptions:
    - LLMParsingException: Custom exception raised when a code block extraction from a message fails.

Static Methods:
    - normalize_snippet(snippet: str) -> List[str]: Normalizes code snippets by removing trailing whitespace
      and empty lines, returning a list of non-empty lines.
    - normalize_code(code: str) -> str: Parses the code using `ast`, then unparses it in an attempt to normalize
      the code.

Constants:
    - NO_EDIT_BEFORE_THIS_LINE (str): Marker placed in the code template to indicate the start of a section
      that should not be modified.
    - NO_EDIT_AFTER_THIS_LINE (str): Marker indicating the end of an uneditable section in the code template.
    - ERROR_CODE_STARTS_WITH (str): Error message format for a code snippet not starting with the expected template.
    - ERROR_CODE_ENDS_WITH (str): Error message format for a code snippet not ending with the expected template.


"""
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
    A class that verifies the structural integrity of given Python code templates and executes code checks.
    This class extends SkillBase and performs normalization, execution, and validation of Python code snippets
    embedded in a context's messages. It ensures that the code adheres to specific editable regions
    constrained by particular comment lines and validates against certain conformity criteria.
    
    Attributes:
        _code_template (str):
             A normalized string representation of the code template with line
            endings and whitespace stripped.
        _code_before (str):
             A string holding the non-editable initial part of the code template.
        _code_before_line_count (int):
             The count of lines in the non-editable initial part of the code template.
        _code_after (str):
             A string holding the non-editable final part of the code template.
        _code_after_line_count (int):
             The count of lines in the non-editable final part of the code template.
    
    Methods:
        __init__(self, code_template:
             str): Initializes the class, normalizes the provided code_template,
            extracts non-editable parts, and validates the full template's stability.
        normalize_snippet(snippet:
             str) -> List[str]: Normalizes a code snippet by removing trailing
            whitespaces and empty lines.
        execute(self, context:
             SkillContext) -> ChatMessage: Processes the last message in the context,
            extracts the Python code, validates it, and
            formats a success or error message accordingly.
        _validate_code(self, code:
             str): Validates that the input code matches the non-editable
            sections of the code template and raises an exception
            with details if validation fails.
        normalize_code(code) -> str:
             Parses the input code into an AST (Abstract Syntax Tree),
            then unparses it back to a normalized code string.

    """

    def __init__(self, code_template: str = ""):
        """
        Initialize the instance with a code template and preprocess it.
        This initializer takes a code template, normalizes it, and segregates it into parts before and after
        specified marker lines. It also validates the stability of the provided code template.
        
        Args:
            code_template (str, optional):
                 A string containing the code template to be processed. Defaults to an empty string.
        
        Raises:
            Exception:
                 If the code template provided is not stable, an exception is raised.
            The __init__ method processes the code template in several steps:
                - Normalize the snippet.
                - Identify and extract the part of the code before the 'NO_EDIT_BEFORE_THIS_LINE' marker.
                - Identify and extract the part of the code after the 'NO_EDIT_AFTER_THIS_LINE' marker.
                - Validate the code template to ensure it doesn't cause errors upon future use.
        
        Attributes initialized include:
            - _code_template:
                 The normalized full code template.
            - _code_before_line_count:
                 The number of lines before the 'NO_EDIT_BEFORE_THIS_LINE' marker.
            - _code_before:
                 The actual code content before the 'NO_EDIT_BEFORE_THIS_LINE' marker.
            - _code_after_line_count:
                 The number of lines after the 'NO_EDIT_AFTER_THIS_LINE' marker.
            - _code_after:
                 The actual code content after the 'NO_EDIT_AFTER_THIS_LINE' marker.
            

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
        """
        Normalizes a given code snippet by splitting it into separate lines, removing
        trailing white spaces, and omitting empty lines.
        
        Args:
            snippet (str):
                 A string representation of the code snippet that needs to be normalized.
        
        Returns:
            (List[str]):
                 A list of strings, where each string is a line from the original snippet
                that has been stripped of trailing white spaces and non-empty.
            

        """
        lines = snippet.splitlines()
        lines = [line.rstrip() for line in lines]
        return [line for line in lines if line != ""]

    def execute(self, context: SkillContext) -> ChatMessage:
        """
        Executes a given skill within a context by parsing the latest chat message for a Python code block and validating it against a predefined template.
        This method takes a `SkillContext` object which encapsulates the information required to execute a skill, including receiving and responding to chat messages.
        It first attempts to extract the last chat message from the given `context`. Then, it searches for a Python code block within the message. If the extracted code is found to be identical to a specified code template, an error message is generated. Otherwise, it normalizes and validates the code.
        If the execution is successful, a formatted success message containing the Python code block is returned. If any exceptions occur during the process, a detailed error message is generated and logged, and then returned.
        
        Args:
            context (SkillContext):
                 The execution context containing methods and attributes required for the skill's operation.
        
        Returns:
            (ChatMessage):
                 A message object to be sent back to the chat, indicating success or failure of the code execution.
        
        Raises:
            Exception:
                 If any exception occurs during code block extraction, normalization, validation, or message building, it is caught and processed into an error message.

        """
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

    def _validate_code(self, code: str):
        """
        Validate the given code snippet against predefined code structure.
        This internal method checks if the `code` snippet starts and/or ends with
        given code lines stored in `_code_before` and `_code_after` respectively.
        If the code does not conform to the predefined structure, it compiles
        a list of descriptive errors indicating what was expected and what the
        actual code was and raises an Exception containing these error messages.
        
        Args:
            code (str):
                 A string representing the code snippet to validate.
        
        Raises:
            Exception:
                 If validation fails, an Exception is raised with a detailed
                message of each code structure violation.
            

        """
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
    def normalize_code(code):
        """
        
        Returns the normalized version of the given source code string.
            This method takes a string of Python code, parses it into an Abstract Syntax
            Tree (AST), and then unparses the AST back into a normalized code string
            which includes a trailing newline. This process can standardize the formatting
            of the input code and remove any unnecessary variations in whitespace, comments,
            or layout, potentially making it more consistent and readable. Type comments
            are preserved during parsing if present.
        
        Args:
            code (str):
                 The Python source code to normalize.
        
        Returns:
            (str):
                 The normalized Python source code, with standardized formatting
                and a trailing newline.
            

        """
        module = ast.parse(code, type_comments=True)
        return ast.unparse(module).strip() + "\n"
