from __future__ import annotations

from council.utils import read_env_str, Parameter, read_env_int, greater_than_validator, prefix_validator
from council.llm.llm_configuration_base import _DEFAULT_TIMEOUT

_env_var_prefix = "ANTHROPIC_"


def _tv(x: float):
    """
    Temperature and Top_p Validators
    Sampling temperature to use, between 0. and 1.
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("must be in the range [0.0..1.0]")


class AnthropicLLMConfiguration:
    """
    Configuration for :class:AnthropicLLM
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
    ):
        """
        Initialize a new instance

        Args:
            model (str): either `claude-2` or `claude-instant-1`. More details https://docs.anthropic.com/claude/reference/selecting-a-model
            api_key (str): the api key
            max_tokens (int): The maximum number of tokens to generate before stopping.
        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=prefix_validator("claude-"))
        self._api_key = Parameter.string(
            name="api_key", required=True, value=api_key, validator=prefix_validator("sk-")
        )
        self._max_tokens = Parameter.int(
            name="max_tokens", required=True, value=max_tokens, validator=greater_than_validator(0)
        )

        self._timeout = Parameter.int(
            name="timeout", required=False, default=_DEFAULT_TIMEOUT, validator=greater_than_validator(0)
        )
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._top_p = Parameter.float(name="top_p", required=False, validator=_tv)
        self._top_k = Parameter.int(name="top_k", required=False, validator=greater_than_validator(0))

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

    def _read_optional_env(self):
        self._temperature.from_env(_env_var_prefix + "LLM_TEMPERATURE")
        self._top_p.from_env(_env_var_prefix + "LLM_TOP_P")
        self._top_k.from_env(_env_var_prefix + "LLM_TOP_K")
        self._timeout.from_env(_env_var_prefix + "LLM_TIMEOUT")

    @staticmethod
    def from_env() -> AnthropicLLMConfiguration:
        api_key = read_env_str(_env_var_prefix + "API_KEY").unwrap()
        model = read_env_str(_env_var_prefix + "LLM_MODEL").unwrap()
        max_tokens = read_env_int(_env_var_prefix + "LLM_MAX_TOKENS", required=False, default=300).unwrap()
        config = AnthropicLLMConfiguration(model=model, api_key=api_key, max_tokens=max_tokens)
        config._read_optional_env()
        return config
