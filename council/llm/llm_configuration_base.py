import abc
from typing import Any

from council.utils.parameter import Parameter

_DEFAULT_TIMEOUT = 30


def _tv(x: float):
    """
    Temperature Validator
    Sampling temperature to use, between 0. and 2.
    """
    if x < 0.0 or x > 2.0:
        raise ValueError("must be in the range [0.0..2.0]")


def _pv(x: float):
    """
    Penalty Validator
    Penalty must be between -2.0 and 2.0
    """
    if x < -2.0 or x > 2.0:
        raise ValueError("must be in the range [-2.0..2.0]")


def _mtv(x: int):
    """
    Max Token Validator
    Must be positive
    """
    if x <= 0:
        raise ValueError("must be positive")


class LLMConfigurationBase(abc.ABC):
    """
    Configuration for OpenAI LLM Chat Completion GPT Model
    """

    def __init__(self):
        self._temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self._max_tokens = Parameter.int(name="max_tokens", required=False, validator=_mtv)
        self._top_p = Parameter.float(name="top_p", required=False)
        self._n = Parameter.int(name="n", required=False, default=1)
        self._presence_penalty = Parameter.float(name="presence_penalty", required=False, validator=_pv)
        self._frequency_penalty = Parameter.float(name="frequency_penalty", required=False, validator=_pv)

    @property
    def temperature(self) -> Parameter[float]:
        """
        temperature settings for the LLM.
        Ranges from 0.0 to 2.0.
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-temperature
        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """
        The model only takes into account the tokens with the highest probability mass.
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-top_p
        """
        return self._top_p

    @property
    def max_tokens(self) -> Parameter[int]:
        """
        Limit on number of tokens
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-max_tokens
        """
        return self._max_tokens

    @property
    def n(self) -> Parameter[int]:
        """
        How many completions to generate for each prompt.
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-n
        """
        return self._n

    @property
    def presence_penalty(self) -> Parameter[float]:
        """
        Positive values penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.
        Number between -2.0 and 2.0
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-presence_penalty
        """
        return self._presence_penalty

    @property
    def frequency_penalty(self) -> Parameter[float]:
        """
        Positive values penalize new tokens based on their existing frequency in the text so far,
        decreasing the model's likelihood to repeat the same line verbatim.
        Number between -2.0 and 2.0
        See: https://platform.openai.com/docs/api-reference/completions/create#completions-create-frequency_penalty
        """
        return self._frequency_penalty

    def read_env(self, env_var_prefix: str):
        self.temperature.from_env(env_var_prefix + "LLM_TEMPERATURE")
        self.max_tokens.from_env(env_var_prefix + "LLM_MAX_TOKENS")
        self.top_p.from_env(env_var_prefix + "LLM_TOP_P")
        self.n.from_env(env_var_prefix + "LLM_N")
        self.presence_penalty.from_env(env_var_prefix + "LLM_PRESENCE_PENALTY")
        self.frequency_penalty.from_env(env_var_prefix + "LLM_FREQUENCY_PENALTY")

    def build_default_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        def add_param(parameter: Parameter):
            if parameter.is_some():
                payload.setdefault(parameter.name, parameter.unwrap())

        add_param(self._temperature)
        add_param(self._max_tokens)
        add_param(self._top_p)
        add_param(self._n)
        add_param(self._presence_penalty)
        add_param(self._frequency_penalty)
        return payload
