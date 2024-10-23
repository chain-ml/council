import unittest

from council.llm.anthropic_llm import AnthropicCostManager
from council.llm.gemini_llm import GeminiCostManager
from council.llm.openai_chat_completions_llm import OpenAICostManager


class TestAnthropicCostManager(unittest.TestCase):
    def setUp(self):
        self.cost_manager = AnthropicCostManager()

    def test_haiku_cost_calculation(self):
        cost_card = self.cost_manager.find_model_costs("claude-3-haiku-20240307")

        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.25)  # $0.25 per 1M tokens for input * 1M tokens = $0.25
        self.assertEqual(completion_cost, 0.625)  # $1.25 per 1M tokens for output * 0.5M tokens = $0.625

        prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
        self.assertEqual(prompt_cost, 0.025)  # $0.25 * 0.1
        self.assertEqual(completion_cost, 0.0625)  # $1.25 * 0.05

    def test_sonnet_cost_calculation(self):
        sonnet_versions = ["claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"]

        for version in sonnet_versions:
            cost_card = self.cost_manager.find_model_costs(version)

            prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
            self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens for input * 1M tokens = $3.00
            self.assertEqual(completion_cost, 7.50)  # $15.00 per 1M tokens for output * 0.5M tokens = $7.50

            prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
            self.assertEqual(prompt_cost, 0.30)  # $3.00 * 0.1
            self.assertEqual(completion_cost, 0.75)  # $15.00 * 0.05

    def test_opus_cost_calculation(self):
        cost_card = self.cost_manager.find_model_costs("claude-3-opus-20240229")

        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 15.00)  # $15.00 per 1M tokens for input * 1M tokens = $15.00
        self.assertEqual(completion_cost, 37.50)  # $75.00 per 1M tokens for output * 0.5M tokens = $37.50

        prompt_cost, completion_cost = cost_card.get_costs(100_000, 50_000)
        self.assertEqual(prompt_cost, 1.50)  # $15.00 * 0.1
        self.assertEqual(completion_cost, 3.75)  # $75.00 * 0.05

    def test_invalid_model(self):
        self.assertIsNone(self.cost_manager.find_model_costs("invalid-model"))

    def test_consumption_units_and_types(self):
        cost_card = self.cost_manager.find_model_costs("claude-3-haiku-20240307")
        consumptions = cost_card.get_consumptions("claude-3-haiku-20240307", 1_000, 1_000)

        for consumption in consumptions:
            self.assertEqual(consumption.unit, "USD")
            self.assertTrue(consumption.kind.startswith("claude-3-haiku-20240307:"))


class TestGeminiCostManager(unittest.TestCase):
    def test_keys_match(self):
        cost_manager = GeminiCostManager(42)

        keys_up_to_128k = set(cost_manager.COSTS_UNDER_128k.keys())
        keys_longer_128k = set(cost_manager.COSTS_OVER_128k.keys())

        self.assertEqual(keys_up_to_128k, keys_longer_128k)

    def test_find_model_costs_under_128k(self):
        cost_manager = GeminiCostManager(100_000)

        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-flash").input, 0.075)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-flash-8b").input, 0.0375)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-pro").input, 1.25)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.0-pro").input, 0.50)

    def test_find_model_costs_over_128k(self):
        cost_manager = GeminiCostManager(150_000)

        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-flash").output, 0.60)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-flash-8b").output, 0.30)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-pro").output, 10.00)
        self.assertEqual(cost_manager.find_model_costs("gemini-1.0-pro").output, 1.50)

    def test_find_model_costs_at_boundary(self):
        cost_manager = GeminiCostManager(128_000)
        # Should use COSTS_UNDER_128k
        self.assertEqual(cost_manager.find_model_costs("gemini-1.5-pro").input, 1.25)

    def test_find_model_costs_invalid_model(self):
        cost_manager = GeminiCostManager(100_000)
        self.assertIsNone(cost_manager.find_model_costs("invalid-model"))

    def test_cost_calculation_under_128k(self):
        cost_manager = GeminiCostManager(100_000)
        cost_card = cost_manager.find_model_costs("gemini-1.5-pro")

        prompt_tokens = 1_000_000
        completion_tokens = 500_000

        prompt_cost, completion_cost = cost_card.get_costs(prompt_tokens, completion_tokens)

        self.assertEqual(prompt_cost, 1.25)  # $1.25 per 1M tokens for input
        self.assertEqual(completion_cost, 2.50)  # $5.00 per 1M tokens for output * 0.5M tokens

    def test_cost_calculation_over_128k(self):
        cost_manager = GeminiCostManager(200_000)
        cost_card = cost_manager.find_model_costs("gemini-1.5-flash")

        prompt_tokens = 1_000_000
        completion_tokens = 500_000

        prompt_cost, completion_cost = cost_card.get_costs(prompt_tokens, completion_tokens)

        self.assertEqual(prompt_cost, 0.15)  # $0.15 per 1M tokens for input
        self.assertEqual(completion_cost, 0.30)  # $0.60 per 1M tokens for output * 0.5M tokens


class TestOpenAICostManager(unittest.TestCase):
    def setUp(self):
        self.cost_manager = OpenAICostManager()

    def test_gpt35_turbo_cost_calculations(self):
        cost_card = self.cost_manager.find_model_costs("gpt-3.5-turbo-0125")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.50)  # $0.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.75)  # $1.50 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-3.5-turbo-16k-0613")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 2.00)  # $4.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-3.5-turbo-instruct")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 1.50)  # $1.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 1.00)  # $2.00 per 1M tokens * 0.5M

    def test_gpt4_family_cost_calculations(self):
        cost_card = self.cost_manager.find_model_costs("gpt-4-turbo")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 10.00)  # $10.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 15.00)  # $30.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-4")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 30.00)  # $30.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 30.00)  # $60.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-4-32k")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 60.00)  # $60.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 60.00)  # $120.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-4-vision-preview")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 10.00)  # $10.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 15.00)  # $30.00 per 1M tokens * 0.5M

    def test_gpt4o_family_cost_calculations(self):
        cost_card = self.cost_manager.find_model_costs("gpt-4o")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 2.50)  # $2.50 per 1M tokens * 1M
        self.assertEqual(completion_cost, 5.00)  # $10.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-4o-2024-05-13")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 5.00)  # $5.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 7.50)  # $15.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("gpt-4o-mini")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 0.150)  # $0.150 per 1M tokens * 1M
        self.assertEqual(completion_cost, 0.300)  # $0.60 per 1M tokens * 0.5M

    def test_o1_family_cost_calculations(self):
        cost_card = self.cost_manager.find_model_costs("o1-preview")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 15.00)  # $15.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 30.00)  # $60.00 per 1M tokens * 0.5M

        cost_card = self.cost_manager.find_model_costs("o1-mini")
        prompt_cost, completion_cost = cost_card.get_costs(1_000_000, 500_000)
        self.assertEqual(prompt_cost, 3.00)  # $3.00 per 1M tokens * 1M
        self.assertEqual(completion_cost, 6.00)  # $12.00 per 1M tokens * 0.5M

    def test_invalid_models(self):
        self.assertIsNone(self.cost_manager.find_model_costs("invalid-model"))

        self.assertIsNone(self.cost_manager.find_model_costs("gpt-4-invalid"))
