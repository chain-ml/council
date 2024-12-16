import unittest

from council.llm import CodeBlocksResponseParser, LLMFunction, LLMCostCard
from council.llm.providers.anthropic.anthropic_llm import Usage as AnthropicUsage
from council.llm.providers.anthropic.anthropic_llm_cost import AnthropicConsumptionCalculator
from council.llm.providers.gemini.gemini_llm_cost import GeminiConsumptionCalculator
from council.llm.providers.groq.groq_llm_cost import GroqConsumptionCalculator
from council.llm.providers.openai.openai_llm_cost import OpenAIConsumptionCalculator, Usage as OpenAIUsage
from council.mocks import MockLLM, MockMultipleResponses


def ensure_cost_are_floats(cost_card: LLMCostCard) -> None:
    assert isinstance(cost_card.input, float)
    assert isinstance(cost_card.output, float)


class TestAnthropicConsumptionCalculator(unittest.TestCase):
    def test_all_costs_are_floats(self):
        calculator = AnthropicConsumptionCalculator("model")
        for cost_card_mapping in [calculator.COSTS, calculator.COSTS_CACHING]:
            for cost_card in cost_card_mapping.values():
                ensure_cost_are_floats(cost_card)

    def test_all_cache_models_have_base_costs(self):
        calculator = AnthropicConsumptionCalculator("model")
        for model in calculator.COSTS_CACHING.keys():
            self.assertIn(model, calculator.COSTS)

    def test_haiku_3_cost_calculation(self):
        cost_card = AnthropicConsumptionCalculator("claude-3-haiku-20240307").find_model_costs()

        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.25)  # $0.25 per 1M tokens for input * 1M tokens = $0.25
        self.assertEqual(completion_cost, 0.625)  # $1.25 per 1M tokens for output * 0.5M tokens = $0.625

        prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
        self.assertEqual(prompt_cost, 0.025)  # $0.25 * 0.1
        self.assertEqual(completion_cost, 0.0625)  # $1.25 * 0.05

    def test_haiku_35_cost_calculation(self):
        cost_card = AnthropicConsumptionCalculator("claude-3-5-haiku-20241022").find_model_costs()

        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 1.00)  # $1.00 per 1M tokens for input * 1M tokens = $1.00
        self.assertEqual(completion_cost, 2.50)  # $5.00 per 1M tokens for output * 0.5M tokens = $2.50

        prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
        self.assertEqual(prompt_cost, 0.10)  # $1.00 * 0.1
        self.assertEqual(completion_cost, 0.25)  # $5.00 * 0.05

    def test_haiku_3_cache_cost_calculation(self):
        consumptions = AnthropicConsumptionCalculator("claude-3-haiku-20240307").get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=100_000,
                completion_tokens=50_000,
                cache_creation_prompt_tokens=1_000_000,
                cache_read_prompt_tokens=500_000,
            )
        )

        cache_creation_cost = next(c for c in consumptions if "cache_creation_prompt_tokens_cost" in c.kind)
        cache_read_cost = next(c for c in consumptions if "cache_read_prompt_tokens_cost" in c.kind)

        self.assertEqual(cache_creation_cost.value, 0.30)  # $0.30 per 1M tokens * 1M tokens
        self.assertEqual(cache_read_cost.value, 0.015)  # $0.03 per 1M tokens * 0.5M tokens

    def test_haiku_35_cache_cost_calculation(self):
        consumptions = AnthropicConsumptionCalculator("claude-3-5-haiku-20241022").get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=100_000,
                completion_tokens=50_000,
                cache_creation_prompt_tokens=1_000_000,
                cache_read_prompt_tokens=500_000,
            )
        )

        cache_creation_cost = next(c for c in consumptions if "cache_creation_prompt_tokens_cost" in c.kind)
        cache_read_cost = next(c for c in consumptions if "cache_read_prompt_tokens_cost" in c.kind)

        self.assertEqual(cache_creation_cost.value, 1.25)  # $1.25 per 1M tokens * 1M tokens
        self.assertEqual(cache_read_cost.value, 0.05)  # $0.10 per 1M tokens * 0.5M tokens

    def test_sonnet_cost_calculation(self):
        sonnet_versions = ["claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"]

        for version in sonnet_versions:
            cost_card = AnthropicConsumptionCalculator(version).find_model_costs()

            prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
            self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens for input * 1M tokens = $3.00
            self.assertEqual(completion_cost, 7.50)  # $15.00 per 1M tokens for output * 0.5M tokens = $7.50

            prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
            self.assertEqual(prompt_cost, 0.30)  # $3.00 * 0.1
            self.assertEqual(completion_cost, 0.75)  # $15.00 * 0.05

    def test_sonnet_cache_cost_calculation(self):
        sonnet_versions = ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"]

        for version in sonnet_versions:
            consumptions = AnthropicConsumptionCalculator(version).get_cost_consumptions(
                AnthropicUsage(
                    prompt_tokens=100_000,
                    completion_tokens=50_000,
                    cache_creation_prompt_tokens=1_000_000,
                    cache_read_prompt_tokens=500_000,
                )
            )

            cache_creation_cost = next(c for c in consumptions if "cache_creation_prompt_tokens_cost" in c.kind)
            cache_read_cost = next(c for c in consumptions if "cache_read_prompt_tokens_cost" in c.kind)

            self.assertEqual(cache_creation_cost.value, 3.75)  # $3.75 per 1M tokens * 1M tokens
            self.assertEqual(cache_read_cost.value, 0.15)  # $0.30 per 1M tokens * 0.5M tokens

    def test_opus_cost_calculation(self):
        cost_card = AnthropicConsumptionCalculator("claude-3-opus-20240229").find_model_costs()

        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 15.00)  # $15.00 per 1M tokens for input * 1M tokens = $15.00
        self.assertEqual(completion_cost, 37.50)  # $75.00 per 1M tokens for output * 0.5M tokens = $37.50

        prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
        self.assertEqual(prompt_cost, 1.50)  # $15.00 * 0.1
        self.assertEqual(completion_cost, 3.75)  # $75.00 * 0.05

    def test_opus_cache_cost_calculation(self):
        consumptions = AnthropicConsumptionCalculator("claude-3-opus-20240229").get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=100_000,
                completion_tokens=50_000,
                cache_creation_prompt_tokens=1_000_000,
                cache_read_prompt_tokens=500_000,
            )
        )

        cache_creation_cost = next(c for c in consumptions if "cache_creation_prompt_tokens_cost" in c.kind)
        cache_read_cost = next(c for c in consumptions if "cache_read_prompt_tokens_cost" in c.kind)

        self.assertEqual(cache_creation_cost.value, 18.75)  # $18.75 per 1M tokens * 1M tokens
        self.assertEqual(cache_read_cost.value, 0.75)  # $1.50 per 1M tokens * 0.5M tokens

    def test_invalid_model(self):
        self.assertIsNone(AnthropicConsumptionCalculator("invalid-model").find_model_costs())

    def test_invalid_model_cache_costs(self):
        # doesn't support caching
        consumptions = AnthropicConsumptionCalculator("claude-3-sonnet-20240229").get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=100_000,
                completion_tokens=50_000,
                cache_creation_prompt_tokens=1_000_000,
                cache_read_prompt_tokens=500_000,
            )
        )

        self.assertEqual(len(consumptions), 0)

    def test_consumption_units_and_types(self):
        model = "claude-3-haiku-20240307"
        calculator = AnthropicConsumptionCalculator(model)
        consumptions = calculator.get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=1_000, completion_tokens=1_000, cache_creation_prompt_tokens=0, cache_read_prompt_tokens=0
            )
        )

        for consumption in consumptions:
            self.assertEqual(consumption.unit, "USD")
            self.assertTrue(consumption.kind.startswith(f"{model}:"))

    def test_cache_consumption_units_and_types(self):
        model = "claude-3-5-sonnet-20241022"
        consumptions = AnthropicConsumptionCalculator(model).get_cost_consumptions(
            AnthropicUsage(
                prompt_tokens=100, completion_tokens=50, cache_creation_prompt_tokens=1000, cache_read_prompt_tokens=500
            )
        )

        for consumption in consumptions:
            self.assertEqual(consumption.unit, "USD")
            self.assertTrue(consumption.kind.startswith(f"{model}:"))
            self.assertTrue(consumption.kind.endswith("_cost"))


class TestGeminiConsumptionCalculator(unittest.TestCase):
    def test_all_costs_are_floats(self):
        calculator = GeminiConsumptionCalculator("model", 42)
        for cost_card_mapping in [calculator.COSTS_UNDER_128k, calculator.COSTS_OVER_128k]:
            for cost_card in cost_card_mapping.values():
                ensure_cost_are_floats(cost_card)

    def test_keys_match(self):
        calculator = GeminiConsumptionCalculator("model", 42)

        keys_up_to_128k = set(calculator.COSTS_UNDER_128k.keys())
        keys_longer_128k = set(calculator.COSTS_OVER_128k.keys())

        self.assertEqual(keys_up_to_128k, keys_longer_128k)

    def test_find_model_costs_under_128k(self):
        n = 100_000

        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-flash", n).find_model_costs().input, 0.075)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-flash-8b", n).find_model_costs().input, 0.0375)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-pro", n).find_model_costs().input, 1.25)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.0-pro", n).find_model_costs().input, 0.50)

    def test_find_model_costs_over_128k(self):
        n = 150_000

        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-flash", n).find_model_costs().output, 0.60)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-flash-8b", n).find_model_costs().output, 0.30)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.5-pro", n).find_model_costs().output, 10.00)
        self.assertEqual(GeminiConsumptionCalculator("gemini-1.0-pro", n).find_model_costs().output, 1.50)

    def test_find_model_costs_at_boundary(self):
        calculator = GeminiConsumptionCalculator("gemini-1.5-pro", 128_000)
        # Should use COSTS_UNDER_128k
        self.assertEqual(calculator.find_model_costs().input, 1.25)

    def test_find_model_costs_invalid_model(self):
        calculator = GeminiConsumptionCalculator("invalid-model", 100_000)
        self.assertIsNone(calculator.find_model_costs())

    def test_cost_calculation_under_128k(self):
        calculator = GeminiConsumptionCalculator("gemini-1.5-pro", 100_000)
        cost_card = calculator.find_model_costs()

        prompt_tokens = 1_000_000
        completion_tokens = 500_000

        prompt_cost, completion_cost = cost_card.get_costs(prompt_tokens, completion_tokens)

        self.assertEqual(prompt_cost, 1.25)  # $1.25 per 1M tokens for input
        self.assertEqual(completion_cost, 2.50)  # $5.00 per 1M tokens for output * 0.5M tokens

    def test_cost_calculation_over_128k(self):
        calculator = GeminiConsumptionCalculator("gemini-1.5-flash", 200_000)
        cost_card = calculator.find_model_costs()

        prompt_tokens = 1_000_000
        completion_tokens = 500_000

        prompt_cost, completion_cost = cost_card.get_costs(prompt_tokens, completion_tokens)

        self.assertEqual(prompt_cost, 0.15)  # $0.15 per 1M tokens for input
        self.assertEqual(completion_cost, 0.30)  # $0.60 per 1M tokens for output * 0.5M tokens


class TestOpenAIConsumptionCalculator(unittest.TestCase):
    def test_all_costs_are_floats(self):
        calculator = OpenAIConsumptionCalculator("model")
        for cost_card_mapping in [
            calculator.COSTS_gpt_35_turbo_FAMILY,
            calculator.COSTS_gpt_4_FAMILY,
            calculator.COSTS_gpt_4o_FAMILY,
            calculator.COSTS_o1_FAMILY,
        ]:
            for cost_card in cost_card_mapping.values():
                ensure_cost_are_floats(cost_card)

    def test_gpt35_turbo_cost_calculations(self):
        cost_card = OpenAIConsumptionCalculator("gpt-3.5-turbo-0125").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.50)  # $0.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.75)  # $1.50 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-3.5-turbo-16k-0613").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 2.00)  # $4.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-3.5-turbo-instruct").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 1.50)  # $1.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 1.00)  # $2.00 per 1M tokens * 0.5M

    def test_gpt4_family_cost_calculations(self):
        cost_card = OpenAIConsumptionCalculator("gpt-4-turbo").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 10.00)  # $10.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 15.00)  # $30.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-4").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 30.00)  # $30.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 30.00)  # $60.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-4-32k").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 60.00)  # $60.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 60.00)  # $120.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-4-vision-preview").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 10.00)  # $10.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 15.00)  # $30.00 per 1M tokens * 0.5M

    def test_gpt4o_family_cost_calculations(self):
        cost_card = OpenAIConsumptionCalculator("gpt-4o").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 2.50)  # $2.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 5.00)  # $10.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-4o-2024-05-13").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 5.00)  # $5.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 7.50)  # $15.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("gpt-4o-mini").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.150)  # $0.150 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.300)  # $0.60 per 1M tokens * 0.5M

    def test_o1_family_cost_calculations(self):
        cost_card = OpenAIConsumptionCalculator("o1-preview").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 15.00)  # $15.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 30.00)  # $60.00 per 1M tokens * 0.5M

        cost_card = OpenAIConsumptionCalculator("o1-mini").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 6.00)  # $12.00 per 1M tokens * 0.5M

    def test_invalid_models(self):
        self.assertIsNone(OpenAIConsumptionCalculator("invalid-model").find_model_costs())
        self.assertIsNone(OpenAIConsumptionCalculator("gpt-4-invalid").find_model_costs())

    def test_cached_tokens_cost_calculations(self):
        usage = OpenAIUsage(
            completion_tokens=0,
            prompt_tokens=500_000,
            total_tokens=1_500_000,
            reasoning_tokens=0,
            cached_tokens=1_000_000,
        )

        consumptions = OpenAIConsumptionCalculator("gpt-4o").get_cost_consumptions(usage)

        cached_cost = next(c.value for c in consumptions if "cache_read_prompt_tokens_cost" in c.kind)
        prompt_cost = next(c.value for c in consumptions if "prompt_tokens_cost" in c.kind and "cache" not in c.kind)

        # Cached tokens should cost half of the normal prompt rate for gpt-4: $2.50 per 1M tokens for input -> 1.25
        self.assertEqual(cached_cost, 1.25)

        # Regular prompt tokens (500k) at full rate -> 1.25
        self.assertEqual(prompt_cost, 1.25)

    def test_reasoning_tokens_cost_calculations(self):
        usage = OpenAIUsage(
            completion_tokens=1_000_000,
            prompt_tokens=0,
            total_tokens=2_000_000,
            reasoning_tokens=1_000_000,
            cached_tokens=0,
        )

        consumptions = OpenAIConsumptionCalculator("o1-mini").get_cost_consumptions(usage)

        reasoning_cost = next(c.value for c in consumptions if "reasoning_tokens_cost" in c.kind)
        completion_cost = next(c.value for c in consumptions if "completion_tokens_cost" in c.kind)

        self.assertEqual(reasoning_cost, 12.00)
        self.assertEqual(completion_cost, 12.00)


class TestGroqConsumptionCalculator(unittest.TestCase):
    def test_all_costs_are_floats(self):
        calculator = GroqConsumptionCalculator("model")
        for cost_card in calculator.COSTS.values():
            ensure_cost_are_floats(cost_card)

    def test_gemma_cost_calculations(self):
        cost_card = GroqConsumptionCalculator("gemma2-9b-it").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.20)  # $0.20 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.10)  # $0.20 per 1M tokens * 0.5M

        cost_card = GroqConsumptionCalculator("gemma-7b-it").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.07)  # $0.07 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.035)  # $0.07 per 1M tokens * 0.5M

    def test_llama3_cost_calculations(self):
        cost_card = GroqConsumptionCalculator("llama3-70b-8192").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.59)  # $0.59 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.395)  # $0.79 per 1M tokens * 0.5M

        cost_card = GroqConsumptionCalculator("llama3-8b-8192").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.05)  # $0.05 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.04)  # $0.08 per 1M tokens * 0.5M

    def test_llama3_tool_preview_cost_calculations(self):
        cost_card = GroqConsumptionCalculator("llama3-groq-70b-8192-tool-use-preview").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.89)  # $0.89 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.445)  # $0.89 per 1M tokens * 0.5M

        cost_card = GroqConsumptionCalculator("llama3-groq-8b-8192-tool-use-preview").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.19)  # $0.19 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.095)  # $0.19 per 1M tokens * 0.5M

    def test_llama_31_cost_calculations(self):
        cost_card = GroqConsumptionCalculator("llama-3.1-70b-versatile").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.59)  # $0.59 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.395)  # $0.79 per 1M tokens * 0.5M

        cost_card = GroqConsumptionCalculator("llama-3.1-8b-instant").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.05)  # $0.05 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.04)  # $0.08 per 1M tokens * 0.5M

    def test_llama_32_cost_calculations(self):
        models = {
            "llama-3.2-1b-preview": (0.04, 0.04),
            "llama-3.2-3b-preview": (0.06, 0.06),
            "llama-3.2-11b-vision-preview": (0.18, 0.18),
            "llama-3.2-90b-vision-preview": (0.90, 0.90),
        }

        for model, (input_cost, output_cost) in models.items():
            cost_card = GroqConsumptionCalculator(model).find_model_costs()
            prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
            self.assertEqual(prompt_cost, input_cost)  # input_cost per 1M tokens * 1M
            self.assertEqual(completion_cost, output_cost * 0.5)  # output_cost per 1M tokens * 0.5M

    def test_others(self):
        cost_card = GroqConsumptionCalculator("llama-guard-3-8b").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.20)  # $0.20 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.10)  # $0.20 per 1M tokens * 0.5M

        cost_card = GroqConsumptionCalculator("mixtral-8x7b-32768").find_model_costs()
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.24)  # $0.24 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.12)  # $0.24 per 1M tokens * 0.5M

    def test_invalid_model(self):
        self.assertIsNone(GroqConsumptionCalculator("invalid-model").find_model_costs())


class Response(CodeBlocksResponseParser):
    block: str


class TestSelfCorrectionConsumptions(unittest.TestCase):
    incorrect_response = "incorrect"
    correct_response = """
```block
correct
```
"""
    responses = [[incorrect_response], [correct_response]]

    def test_self_correction_consumptions(self):
        llm = MockLLM(action=MockMultipleResponses(responses=self.responses))
        llm_func = LLMFunction(llm, Response.from_response, system_message="")

        llm_func_response = llm_func.execute_with_llm_response(user_message="")

        assert len(llm_func_response.consumptions) == 2  # two "call" consumptions
