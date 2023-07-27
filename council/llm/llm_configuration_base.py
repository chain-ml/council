import abc
from typing import Any

from council.utils.parameter import Parameter


def _tv(x: float):
    """
    Temperature Validator
    Sampling temperature to use, between 0. and 2.
    """
    if x < 0.0 or x > 2.0:
        raise Exception("must be in the range [0.0..2.0]")


def _pv(x: float):
    """
    Penalty Validator
    Penalty must be between -2.0 and 2.0
    """
    if x < -2.0 or x > 2.0:
        raise Exception("must be in the range [-2.0..2.0]")


def _mtv(x: int):
    """
    Max Token Validator
    Must be positive
    """
    if x <= 0:
        raise Exception("must be positive")


class LLMConfigurationBase(abc.ABC):
    """
    Configuration for OpenAI LLM Chat Completion GPT Model

    Args:
        temperature (float): optional temperature settings for the LLM
        max_tokens (int): optional limit on number of tokens
        top_p (int): optional he model only takes into account the tokens with the highest probability mass
        n (int): optional How many completions to generate for each prompt
        presence_penalty (float): optional, impacts how the model penalizes new tokens based on whether
            they have appeared in the text so far
        frequency_penalty (float): optional, impacts how the model penalizes new tokens based on their existing
            frequency in the text.
    """

    temperature: Parameter[float]
    max_tokens: Parameter[int]
    top_p: Parameter[float]
    n: Parameter[int]
    presence_penalty: Parameter[float]
    frequency_penalty: Parameter[float]

    def __init__(self):
        self.temperature = Parameter.float(name="temperature", required=False, default=0.0, validator=_tv)
        self.max_tokens = Parameter.int(name="max_tokens", required=False, validator=_mtv)
        self.top_p = Parameter.float(name="top_p", required=False)
        self.n = Parameter.int(name="n", required=False, default=1)
        self.presence_penalty = Parameter.float(name="presence_penalty", required=False, validator=_pv)
        self.frequency_penalty = Parameter.float(name="frequency_penalty", required=False, validator=_pv)

    def read_env(self, env_var_prefix: str):
        self.temperature.from_env(env_var_prefix + "LLM_TEMPERATURE")
        self.max_tokens.from_env(env_var_prefix + "LLM_MAX_TOKENS")
        self.top_p.from_env(env_var_prefix + "LLM_TOP_P")
        self.n.from_env(env_var_prefix + "LLM_N")
        self.presence_penalty.from_env(env_var_prefix + "LLM_PRESENCE_PENALTY")
        self.frequency_penalty.from_env(env_var_prefix + "LLM_FREQUENCY_PENALTY")

    def build_default_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.temperature.is_some():
            payload.setdefault(self.temperature.name, self.temperature.unwrap())
        if self.max_tokens.is_some():
            payload.setdefault(self.max_tokens.name, self.max_tokens.unwrap())
        if self.top_p.is_some():
            payload.setdefault(self.top_p.name, self.top_p.unwrap())
        if self.n.is_some():
            payload.setdefault(self.n.name, self.n.unwrap())
        if self.presence_penalty.is_some():
            payload.setdefault(self.presence_penalty.name, self.presence_penalty.unwrap())
        if self.frequency_penalty.is_some():
            payload.setdefault(self.frequency_penalty.name, self.frequency_penalty.unwrap())
        return payload
