from council.llm.llm_answer import LLMParsingException
from council.utils import CodeParser


def extract_code_block(text: str, block_type: str = "") -> str:
    result = CodeParser.find_first(block_type, text)
    if result is None:
        if block_type == "":
            raise LLMParsingException("could not find a code block")
        else:
            raise LLMParsingException(f"could not find a code block of type `{block_type}`")
    return result.code
