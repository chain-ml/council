from __future__ import annotations
from typing import Optional

from council.utils import read_env_str


class AnthropicLLMConfiguration:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__()
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        config = AnthropicLLMConfiguration()

        config.model = read_env_str("ANTHROPIC_MODEL").unwrap()
        config.api_key = read_env_str("ANTHROPIC_API_KEY").unwrap()

        return config
