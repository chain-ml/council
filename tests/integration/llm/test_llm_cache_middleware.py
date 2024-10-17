import time
import unittest

import dotenv

from council import AnthropicLLM
from council.llm.llm_function import LLMFunction
from council.llm.llm_middleware import LLMCachingMiddleware, LLMResponse
from council.utils import OsEnviron


class TestLlmCachingMiddleware(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            self.llm = AnthropicLLM.from_env()

        self.llm_func: LLMFunction[LLMResponse] = LLMFunction(
            self.llm, lambda response: response, system_message="You're a helpful assistant"
        )

        self.llm_func.add_middleware(LLMCachingMiddleware(ttl=60))

    def execute_llm_func(self, message: str, to_print: str, **kwargs) -> None:
        start_time = time.time()
        first_response = self.llm_func.execute(message, **kwargs)
        wall_time = time.time() - start_time
        print(f"\n{to_print}")
        print(f"\tResponse duration: {first_response.duration:.3f}s")
        print(f"\tWall time: {wall_time:.3f}s")

    def test_caching(self):
        # assert on wall_time
        self.execute_llm_func("What is the capital of France?", to_print="First request")
        self.execute_llm_func("What is the capital of France?", to_print="Second request (cached)")
        self.execute_llm_func("What is the capital of France???", to_print="Third request (different input)")
        self.execute_llm_func(
            "What is the capital of France?", to_print="Fourth request (different params)", temperature=0.9
        )
        self.execute_llm_func("What is the capital of France?", to_print="Fifth request (cached)", temperature=0.9)
