from __future__ import annotations

from council.utils import read_env_str, Parameter, read_env_int
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

env_var_prefix = "ANTHROPIC_"


def _model_validator(x: str):
    if x != "claude-2" and x != "claude-instant-1":
        raise ValueError("must be either `claude-2` or `claude-instant-1`")


def _tv(x: float):
    """
    Temperature and Top_p Validators
    Sampling temperature to use, between 0. and 1.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("must be in the range [0.0..1.0]")


def _positive_validator(x: int):
    """
    Max Token and Top_K Validator
    Must be positive
    """
    if x <= 0:
        raise ValueError("must be positive")


class AnthropicLLMConfiguration:
    """
    Configuration for Anthropic LLMs
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
        timeout: int = _DEFAULT_TIMEOUT,
        temperature: float = 0.0,
    ):
        """
        Initialize a new instance

        Args:
            model (str): either `claude-2` or `claude-instant-1`. More details https://docs.anthropic.com/claude/reference/selecting-a-model
            api_key (str): the api key
        """

        super().__init__()
        self.model = Parameter.string("model", required=True, validator=_model_validator)
        self.model.set(model)

        self.api_key = Parameter.string("model", required=True)
        self.api_key.set(api_key)

        self.timeout = Parameter.int(name="timeout", required=False, default=timeout, validator=_positive_validator)
        self.temperature = Parameter.float(name="temperature", required=False, default=temperature, validator=_tv)
        self.top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self.top_k = Parameter.int(name="top_k", required=False, validator=_positive_validator)

        self.max_tokens = Parameter.int(name="max_tokens", required=True, validator=_positive_validator)
        self.max_tokens.set(max_tokens)

    def read_optional_env(self):
        self.temperature.from_env(env_var_prefix + "LLM_TEMPERATURE")
        self.top_p.from_env(env_var_prefix + "LLM_TOP_P")
        self.top_k.from_env(env_var_prefix + "LLM_TOP_K")
        self.timeout.from_env(env_var_prefix + "LLM_TIMEOUT")

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        model = read_env_str(env_var_prefix + "MODEL").unwrap()
        api_key = read_env_str(env_var_prefix + "API_KEY").unwrap()
        max_tokens = read_env_int(env_var_prefix + "LLM_MAX_TOKENS", required=False, default=300).unwrap()
        config = AnthropicLLMConfiguration(model=model, api_key=api_key, max_tokens=max_tokens)
        config.read_optional_env()
        return config
