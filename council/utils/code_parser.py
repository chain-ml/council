import re
from typing import Optional, List, Tuple


class CodeParser:
    def __init__(self, language: Optional[str] = None):
        self._pattern = r"```(\w*)\n(.*?)\n```" if language is None else rf"```{language}\n(.*?)\n```"

    def extract_code(self, text: str) -> List[Tuple[str, str]]:
        """Extract code from a text.

        Args:
            text (str): The text to extract code from.

        Returns:
            list: A list of tuples, each containing the language and the code.
              If there is no code block in the input text, the language would be "unknown".
        """
        matches = re.findall(self._pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                bloc = match
