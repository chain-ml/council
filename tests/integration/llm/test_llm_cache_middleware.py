import time
import unittest

import dotenv

from council import AnthropicLLM
from council.llm.llm_function import LLMFunction
from council.llm.llm_middleware import LLMCachingMiddleware, LLMResponse
from council.llm.llm_response_parser import EchoResponseParser
from council.utils import OsEnviron


class TestLlmCachingMiddleware(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            self.llm = AnthropicLLM.from_env()

        self.llm_func: LLMFunction[LLMResponse] = LLMFunction(
            self.llm, EchoResponseParser.from_response, system_message="You're a helpful assistant"
        )

        self.llm_func.add_middleware(LLMCachingMiddleware(ttl=60))

    def get_llm_func_wall_time(self, message: str, to_print: str, **kwargs) -> float:
        start_time = time.time()
        first_response = self.llm_func.execute(message, **kwargs)
        wall_time = time.time() - start_time
        print(f"\n{to_print}")
        print(f"\tResponse duration: {first_response.duration:.3f}s")
        print(f"\tWall time: {wall_time:.3f}s")

        return wall_time

    def test_caching(self):
        message_v1 = "What is the capital of France?"
        message_v2 = "What is the capital of France???"
        kwargs = {"temperature": 0.9}

        _ = self.get_llm_func_wall_time(message_v1, to_print="First request")
        t = self.get_llm_func_wall_time(message_v1, to_print="Second request (cached)")
        assert t < 0.1

        _ = self.get_llm_func_wall_time(message_v2, to_print="Third request (different input)")
        _ = self.get_llm_func_wall_time(message_v1, to_print="Fourth request (different params)", **kwargs)
        t = self.get_llm_func_wall_time(message_v1, to_print="Fifth request (cached)", **kwargs)
        assert t < 0.1
