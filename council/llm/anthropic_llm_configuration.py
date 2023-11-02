from __future__ import annotations

from council.utils import read_env_str, Parameter, read_env_int
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "ANTHROPIC_"


def _model_validator(x: str):
    if not x.startswith("claude-2") and not x.startswith("claude-instant-1"):
        raise ValueError("base must be either `claude-2` or `claude-instant-1`")


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
        self._model = Parameter.string(name="model", required=True, value=model, validator=_model_validator)
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key)
        self._timeout = Parameter.int(name="timeout", required=False, default=timeout, validator=_positive_validator)
        self._temperature = Parameter.float(name="temperature", required=False, default=temperature, validator=_tv)
        self._top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self._top_k = Parameter.int(name="top_k", required=False, validator=_positive_validator)
        self._max_tokens = Parameter.int(
            name="max_tokens", required=True, value=max_tokens, validator=_positive_validator
        )

    def read_optional_env(self):
        self._temperature.from_env(_env_var_prefix + "LLM_TEMPERATURE")
        self._top_p.from_env(_env_var_prefix + "LLM_TOP_P")
        self._top_k.from_env(_env_var_prefix + "LLM_TOP_K")
        self._timeout.from_env(_env_var_prefix + "LLM_TIMEOUT")

    @property
    def model(self) -> Parameter[str]:
        """
        Anthropic model
        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        Anthropic API Key
        """
        return self._api_key

    @property
    def timeout(self) -> Parameter[int]:
        """
        API timeout
        """
        return self._timeout

    @property
    def temperature(self) -> Parameter[float]:
        """
        Amount of randomness injected into the response.
        Ranges from 0 to 1.
        Use temp closer to 0 for analytical / multiple choice, and closer to 1 for creative and generative tasks.
        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """
        Use nucleus sampling.
        In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in
        decreasing probability order and cut it off once it reaches a particular probability specified by top_p.
        """
        return self._top_p

    @property
    def top_k(self) -> Parameter[int]:
        """
        Only sample from the top K options for each subsequent token.
        Used to remove "long tail" low probability responses.
        """
        return self._top_k

    @property
    def max_tokens(self) -> Parameter[int]:
        """
        The maximum number of tokens to generate before stopping.
        Note that models may stop before reaching this maximum.
        This parameter only specifies the absolute maximum number of tokens to generate.
        """
        return self._max_tokens

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        model = read_env_str(_env_var_prefix + "MODEL").unwrap()
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        max_tokens = read_env_int(_env_var_prefix + "LLM_MAX_TOKENS", required=False, default=300).unwrap()
        config = AnthropicLLMConfiguration(model=model, api_key=api_key, max_tokens=max_tokens)
        config.read_optional_env()
        return config
