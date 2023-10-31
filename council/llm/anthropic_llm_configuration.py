from __future__ import annotations
from typing import Optional

from council.utils import read_env_str


class AnthropicLLMConfiguration:
    """
    Configuration for Anthropic LLMs
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize a new instance

        Args:
            model(Optional[str]): either `claude-2` or `claude-instant-1`. More details https://docs.anthropic.com/claude/reference/selecting-a-model
            api_key(Optional[str]): the api key
        """

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
