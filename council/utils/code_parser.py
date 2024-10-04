from __future__ import annotations

from typing import Iterable, List, Optional

from more_itertools import first, last


class CodeBlock:
    def __init__(self, language: Optional[str], code: str) -> None:
        self._language = language
        self._code = code

    @property
    def code(self) -> str:
        return self._code

    @property
    def language(self) -> Optional[str]:
        return self._language

    def is_language(self, language: str) -> bool:
        if self._language is None:
            return False
        return self._language.casefold() == language.casefold()


class CodeParser:
    DELIMITER = "```"

    @staticmethod
    def iter_code_blocs(language: Optional[str] = None, text: str = "") -> Iterable[CodeBlock]:
        return CodeParser._build_generator(language, text)

    @staticmethod
    def extract_code_blocs(language: Optional[str] = None, text: str = "") -> List[CodeBlock]:
        return list(CodeParser._build_generator(language, text))

    @staticmethod
    def find_first(language: Optional[str] = None, text: str = "") -> Optional[CodeBlock]:
        blocks = CodeParser._build_generator(language, text)
        return first(blocks, None)

    @staticmethod
    def find_last(language: Optional[str] = None, text: str = "") -> Optional[CodeBlock]:
        blocks = CodeParser._build_generator(language, text)
        return last(blocks, None)

    @staticmethod
    def _build_generator(language: Optional[str], text: str = "") -> Iterable[CodeBlock]:
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            if not lines[i].startswith(CodeParser.DELIMITER):
                i += 1
                continue

            # start of a code block found
            actual_block_language = lines[i][len(CodeParser.DELIMITER) :].strip() or None
            start_index = i + 1
            i += 1

            while i < len(lines) and lines[i] != CodeParser.DELIMITER:
                i += 1

            if i >= len(lines):
                break  # incomplete block

            if actual_block_language == language or language is None:
                yield CodeBlock(actual_block_language, "\n".join(lines[start_index:i]))

            i += 1  # skip the closing delimiter
