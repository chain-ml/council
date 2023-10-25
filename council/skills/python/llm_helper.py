class LLMMessageParseException(Exception):
    pass


def extract_code_block(text: str, block_type: str = "") -> str:
    start_token = f"```{block_type}\n"
    end_token = "\n```"
    start = text.find(start_token)
    if start == -1:
        if block_type == "":
            raise LLMMessageParseException("could not find a code block")
        else:
            raise LLMMessageParseException(f"could not find a code block of type `{block_type}`")
    start += len(start_token)
    end = text.find(end_token, start)
    if end == -1:
        if block_type == "":
            raise LLMMessageParseException("could not find the end of the code block")
        else:
            raise LLMMessageParseException(f"could not find the end of the code block of type `{block_type}`")
    return text[start:end]
