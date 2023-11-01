from __future__ import annotations
from typing import Optional

from council.utils import read_env_str, read_env_float
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT


class AnthropicLLMConfiguration:
    """
    Configuration for Anthropic LLMs
    """

    timeout: float
    model: str
    api_key: str

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, timeout: Optional[float] = None):
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
        if timeout is not None:
            self.timeout = timeout

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        config = AnthropicLLMConfiguration()

        config.model = read_env_str("ANTHROPIC_MODEL").unwrap()
        config.api_key = read_env_str("ANTHROPIC_API_KEY").unwrap()
        config.timeout = read_env_float(name="ANTHROPIC_LLM_TIMEOUT", required=False, default=_DEFAULT_TIMEOUT).unwrap()

        return config
