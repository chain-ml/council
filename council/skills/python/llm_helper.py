"""

Module to extract code blocks from text using given block type identifiers.

This module consists of functions that help in parsing and extracting code blocks from a given string of text. It leverages the CodeParser class to find the first instance of code block specified by the block type, and raises exceptions if a code block cannot be found.

Raises:
    LLMParsingException: An error thrown when a code block of the desired type is not found within the text.

Functions:
    extract_code_block: Extracts the first code block from the provided text that matches the specified block type.


"""
from council.llm.llm_answer import LLMParsingException
from council.utils import CodeParser


def extract_code_block(text: str, block_type: str = "") -> str:
    """
    Extracts a specific type of code block from the given text.
    This function uses a code parser to find the first occurrence of a code block of the specified type within the provided text. If the specified block type is not found, an exception is raised.
    
    Args:
        text (str):
             The text from which to extract the code block.
        block_type (str, optional):
             The type of code block to extract. Defaults to an empty string, which implies no specific type.
    
    Returns:
        (str):
             The code of the first matching code block found within the text.
    
    Raises:
        LLMParsingException:
             If a code block of the specified type cannot be found.
        

    """
    result = CodeParser.find_first(block_type, text)
    if result is None:
        if block_type == "":
            raise LLMParsingException("could not find a code block")
        else:
            raise LLMParsingException(f"could not find a code block of type `{block_type}`")
    return result.code
