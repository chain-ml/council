from typing import Optional

from council.llm import LLMBase, get_default_llm


def get_test_default_llm(max_retries: Optional[int] = None) -> LLMBase:
    return get_default_llm(max_retries=max_retries)
