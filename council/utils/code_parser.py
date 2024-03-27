"""

A module for parsing blocks of code from text strings. The `CodeBlock` class represents a block of code, including its language (if specified) and the actual code content. The `CodeParser` class provides methods to extract and iterate over `CodeBlock` instances within a given text, based on a specified programming language or any language if none is specified.

Classes:
    CodeBlock: Encapsulation of a single code block with its respective language and code.
    CodeParser: Utility class that offers methods to parse code blocks from text.

Functions:
    CodeParser.iter_code_blocs(language: Optional[str], text: str) -> Iterable[CodeBlock]: Iterates over code blocks detected in a given text.
    CodeParser.extract_code_blocs(language: Optional[str], text: str) -> List[CodeBlock]: Extracts and returns a list of code blocks detected in a given text.
    CodeParser.find_first(language: Optional[str], text: str) -> Optional[CodeBlock]: Finds and returns the first code block matching the specified language in the given text.
    CodeParser.find_last(language: Optional[str], text: str) -> Optional[CodeBlock]: Finds and returns the last code block matching the specified language in the given text.

The module also includes private helper methods for constructing the regular expression pattern to match code blocks and for generating code block instances.


"""
from __future__ import annotations
from more_itertools import first, last
import re
from typing import Optional, List, Iterable


class CodeBlock:
    """
    A class representing a block of code with optional language specification.
    
    Attributes:
        language (str, optional):
             The programming language of the code block. It can be `None` if the language is not specified.
        code (str):
             The actual code contained within the code block.
    
    Methods:
        __init__:
             Initializes a new instance of the CodeBlock class.
        code:
             Property that returns the code contained within the block.
        language:
             Property that returns the programming language of the code block, or `None` if not specified.
        is_language:
             Checks if the code block's language matches the given language (case-insensitive).
    
    Args:
        language (Optional[str]):
             The programming language of the code block. Defaults to `None` if not specified.
        code (str):
             The actual code contained within the code block.

    """
    def __init__(self, language: Optional[str], code: str):
        """
        Initializes a new instance of the class with specified language and code attributes.
        
        Args:
            language (Optional[str]):
                 The programming language of the given code. Can be 'None' if the language is not specified.
            code (str):
                 The string representation of the code to be associated with the instance.
        
        Raises:
            TypeError:
                 If 'code' is not provided as a string.

        """
        self._language = language
        self._code = code

    @property
    def code(self) -> str:
        """
        Gets the code property of the object.
        This property getter returns the value of the private '_code' attribute, encapsulating the access to the internal representation of the code.
        
        Returns:
            (str):
                 The current value of the _code attribute.

        """
        return self._code

    @property
    def language(self) -> Optional[str]:
        """
        Property that gets the language attribute.
        This property is a getter that returns the language setting of an object. If the language has not been
        set, it may return None.
        
        Returns:
            (Optional[str]):
                 The language code as a string. None if the language is not set.
            

        """
        return self._language

    def is_language(self, language: str) -> bool:
        """
        Check if the current object's language matches the specified language.
        The comparison is case-insensitive, meaning that case differences are ignored.
        
        Args:
            language (str):
                 The language to compare with the object's language.
        
        Returns:
            (bool):
                 True if the specified language matches the object's language, otherwise False.
                When the object's language is not set, it returns False by default.
            

        """
        if self._language is None:
            return False
        return self._language.casefold() == language.casefold()


class CodeParser:
    """
    A CodeParser is a utility class for parsing code blocks within a string. It provides static methods to iterate over, extract, and find specific code blocks dictated by a language specifier. The class primarily parses code blocks denoted by markdown code fencing (i.e., text wrapped between triple backticks ```).
    
    Attributes:
        None
    
    Methods:
        iter_code_blocs(language:
             Optional[str]=None, text: str='') -> Iterable[CodeBlock]
            Returns an iterator over CodeBlock objects found in the given text.
        extract_code_blocs(language:
             Optional[str]=None, text: str='') -> List[CodeBlock]
            Extracts and returns a list of CodeBlock objects found in the given text.
        find_first(language:
             Optional[str]=None, text: str='') -> Optional[CodeBlock]
            Finds and returns the first CodeBlock object in the text, or None if no block is found.
        find_last(language:
             Optional[str]=None, text: str='') -> Optional[CodeBlock]
            Finds and returns the last CodeBlock object in the text, or None if no block is found.
        _get_pattern(language:
             Optional[str]) -> str
            Internal method that constructs a regex pattern to match code blocks, optionally filtered by language.
        _build_generator(language:
             Optional[str], text: str='') -> Iterable[CodeBlock]
            Internal method that builds a generator yielding CodeBlock objects by parsing the text using regex matching.
    
    Note:
        CodeBlock is presumed to be a user-defined type representing a code block with its language and content.

    """
    @staticmethod
    def iter_code_blocs(language: Optional[str] = None, text: str = "") -> Iterable[CodeBlock]:
        """
        Iterates through code blocks extracted from text based on the specified language.
        The function statically parses the provided text to extract and yield code blocks. It uses CodeParser's internal _build_generator method to create and return an iterable of CodeBlock objects. If a language is specified, only code blocks that match the language will be considered, otherwise, code blocks of any language present in the text will be yielded.
        
        Args:
            language (Optional[str]):
                 The programming language to filter code blocks by. If None, blocks of any language will be yielded.
            text (str):
                 The text from which to extract code blocks.
        
        Returns:
            (Iterable[CodeBlock]):
                 An iterable of CodeBlock objects representing the identified code blocks in the text.
            

        """
        return CodeParser._build_generator(language, text)

    @staticmethod
    def extract_code_blocs(language: Optional[str] = None, text: str = "") -> List[CodeBlock]:
        """
        Extracts code blocks from a given text based on a specified language.
        This static method looks through the provided text and collects code blocks that
        are defined in the specified programming language. If no language is provided,
        it will extract code blocks regardless of language. The code blocks are returned
        as a list of CodeBlock objects.
        
        Args:
            language (Optional[str]):
                 The programming language filter for code blocks extraction.
                If None, code blocks of all languages will be extracted.
            text (str):
                 The text from which code blocks are to be extracted.
        
        Returns:
            (List[CodeBlock]):
                 A list of CodeBlock objects, representing the extracted code blocks
                from the input text.
            

        """
        return list(CodeParser._build_generator(language, text))

    @staticmethod
    def find_first(language: Optional[str] = None, text: str = "") -> Optional[CodeBlock]:
        """
        
        Returns the first `CodeBlock` instance found in the provided text, filtered by the specified language if provided.
            This method parses the given text looking for code blocks and returns the first one encountered. If a language is specified,
            it will only consider code blocks of that language. If no code blocks are found or if the specified language is not found,
            it will return None.
        
        Args:
            language (Optional[str]):
                 A string specifying the programming language to filter code blocks by. If None, all code blocks are considered.
            text (str):
                 The text to be parsed for code blocks.
        
        Returns:
            (Optional[CodeBlock]):
                 The first code block found that matches the specified language, or any code block if no language is specified.
                Returns None if no code blocks are found or the specified language is not in the text.

        """
        blocs = CodeParser._build_generator(language, text)
        return first(blocs, None)

    @staticmethod
    def find_last(language: Optional[str] = None, text: str = "") -> Optional[CodeBlock]:
        """
        
        Returns the last `CodeBlock` object parsed from the given text with an optional filter by language.
            This method utilizes a generator to parse code blocks from the provided text and returns the last
            one that was found, if any. It can filter out the code blocks by a specified programming language if the
            `language` argument is supplied. If no code blocks are found, or if the `language` does not match any
            code blocks, `None` is returned.
        
        Args:
            language (Optional[str]):
                 The programming language to filter code blocks by. Defaults to `None`,
                meaning no filtering is applied.
            text (str):
                 The text input from which to parse code blocks. Defaults to an empty string.
        
        Returns:
            (Optional[CodeBlock]):
                 The last `CodeBlock` object matched by the optional language filter from the
                given text, or `None` if no matching block is found.

        """
        blocs = CodeParser._build_generator(language, text)
        return last(blocs, None)

    @staticmethod
    def _get_pattern(language: Optional[str]):
        """
        Generate a regex pattern for extracting code blocks from text.
        This static method returns a regex pattern that is used to identify and extract
        code blocks from a given text. The pattern can be customized to target a specific
        programming language or to match any language if none is specified.
        The pattern specifically looks for code blocks that are fenced by triple backticks ```,
        which is a common markdown convention to denote code blocks. The pattern captures
        both the language identifier (if present) and the code within the block.
        
        Args:
            language (Optional[str]):
                 A string representing the specific programming language
                for which to tailor the regex pattern. If None, the pattern
                will match code blocks with any language identifier.
        
        Returns:
            (str):
                 A string containing the regex pattern to identify and capture code blocks.
                If `language` is provided, the pattern will only match blocks for that language.
                Otherwise, it will match code blocks with any language identifier or without one.

        """
        return r"```(\w*) *\n(.*?)\n```" if language is None else rf"```({language})\n(.*?)\n```"

    @staticmethod
    def _build_generator(language: Optional[str], text: str = "") -> Iterable[CodeBlock]:
        """
        Generates an iterable sequence of CodeBlock objects from the provided text.
        This static method parses the given text using a language-specific pattern to identify code blocks within the text.
        Each found code block is encapsulated into a CodeBlock object, which is then yielded one by one, allowing
        for iteration over all discovered code blocks.
        
        Args:
            language (Optional[str]):
                 The programming language identifier used to select the parsing pattern.
                If None, parsing may be performed without language-specific considerations.
            text (str):
                 The input string potentially containing code blocks to parse and yield. Defaults to an empty string.
        
        Returns:
            (Iterable[CodeBlock]):
                 An iterable sequence of CodeBlock objects obtained from parsing the input text.
            

        """
        p = CodeParser._get_pattern(language)
        matches = re.finditer(p, text, re.DOTALL)
        for match in matches:
            yield CodeBlock(match.group(1), match.group(2))
