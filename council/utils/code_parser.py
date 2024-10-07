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
        actual_block_language: Optional[str] = None
        code_lines: Optional[List[str]] = None

        for line in text.split("\n"):
            if line.startswith(CodeParser.DELIMITER) and code_lines is None:
                actual_block_language = line[len(CodeParser.DELIMITER) :].strip()
                code_lines = []
                continue

            if line == CodeParser.DELIMITER:
                if (actual_block_language == language or language is None) and code_lines is not None:
                    yield CodeBlock(language, "\n".join(code_lines))
                code_lines = None
                actual_block_language = None
                continue

            if code_lines is not None:
                code_lines.append(line)
