import time
import unittest

import dotenv

from council.llm import AnthropicLLM, EchoResponseParser, LLMFunction, LLMCachingMiddleware, LLMResponse
from council.utils import OsEnviron
from council import OpenAILLM
from council.llm import LLMBase, LLMMessage, LLMRequest, LLMMiddlewareChain
from council.llm.llm_function.llm_middleware import ConfigurationModifierMiddleware, increase_temperature


class TestLlmCachingMiddleware(unittest.TestCase):
    def setUp(self) -> None:
        dotenv.load_dotenv()
        with OsEnviron("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307"):
            self.llm = AnthropicLLM.from_env()

        self.m1 = "What is the capital of France?"
        self.m2 = "What is the capital of France???"
        self.m3 = "What is the capital of France?????"

    def get_llm_func(self, ttl: float, cache_limit_size: int):
        llm_func: LLMFunction[LLMResponse] = LLMFunction(
            self.llm, EchoResponseParser.from_response, system_message="You're a helpful assistant"
        )

        llm_func.add_middleware(LLMCachingMiddleware(ttl=ttl, cache_limit_size=cache_limit_size))
        return llm_func

    @staticmethod
    def execute_llm_func(llm_func: LLMFunction, message: str, to_print: str, **kwargs) -> LLMResponse:
        response = llm_func.execute(message, **kwargs)
        print(f"\n{to_print}")
        print(f"\tResponse duration: {response.duration:.3f}s")
        for consumption in response.result.consumptions:
            print(f"\t{consumption}")

        return response

    @staticmethod
    def raise_if_cached(response):
        assert all(not consumption.unit.startswith("cached_") for consumption in response.result.consumptions)

    @staticmethod
    def raise_if_not_cached(response):
        assert response.duration == 0
        assert all(consumption.unit.startswith("cached_") for consumption in response.result.consumptions)

    def test_caching(self):
        llm_func = self.get_llm_func(ttl=60, cache_limit_size=10)
        kwargs = {"temperature": 0.9}

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1")
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 (cached)")
        self.raise_if_not_cached(response)

        response = self.execute_llm_func(llm_func, self.m2, to_print="message_v2")
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 + kwargs", **kwargs)
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 + kwargs (cached)", **kwargs)
        self.raise_if_not_cached(response)

    def test_ttl(self):
        llm_func = self.get_llm_func(ttl=2, cache_limit_size=10)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1")
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 (cached)")
        self.raise_if_not_cached(response)

        time.sleep(2.5)
        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 after 2s sleep (should be expired)")
        self.raise_if_cached(response)

    def test_cache_limit_size(self):
        llm_func = self.get_llm_func(ttl=60, cache_limit_size=2)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1")
        # cache: m1 <- latest
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m2, to_print="message_v2")
        # cache: m1, m2
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m3, to_print="message_v3")
        # cache: m2, m3
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m3, to_print="message_v3 (cached)")
        # cache: m2, m3
        self.raise_if_not_cached(response)

        response = self.execute_llm_func(llm_func, self.m2, to_print="message_v2 (cached)")
        # cache: m3, m2
        self.raise_if_not_cached(response)

        response = self.execute_llm_func(llm_func, self.m1, to_print="message_v1 (not cached due to size limits)")
        # cache: m2, m1
        self.raise_if_cached(response)

        response = self.execute_llm_func(llm_func, self.m3, to_print="message_v3 (not cached due to size limits)")
        # cache: m1, m3
        self.raise_if_cached(response)


class TestConfigurationModifierMiddleware(unittest.TestCase):
    @staticmethod
    def get_llm() -> LLMBase:
        dotenv.load_dotenv()
        with OsEnviron("OPENAI_LLM_TEMPERATURE", "0.1"):
            return OpenAILLM.from_env()

    def test_increase_temperature(self):
        messages = [LLMMessage.user_message("What is the capital of France?")]
        request = LLMRequest.default(messages)

        llm = self.get_llm()

        self.assertAlmostEqual(llm.configuration.temperature.value, 0.1)
        with_temperature_increase = LLMMiddlewareChain(llm)
        with_temperature_increase.add_middleware(ConfigurationModifierMiddleware(increase_temperature(0.314)))
        _ = with_temperature_increase.execute(request)
        self.assertAlmostEqual(llm.configuration.temperature.value, 0.414)

        _ = with_temperature_increase.execute(request)
        self.assertAlmostEqual(llm.configuration.temperature.value, 0.728)
