import abc
from typing import Any

from council.utils import *


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

    temperature: Option[float]
    max_tokens: Option[int] = Option.none()
    top_p: Option[float] = Option.none()
    n: Option[int] = Option.none()
    presence_penalty: Option[float]
    frequency_penalty: Option[float]

    def __init__(self, env_var_prefix: str):
        self._prefix = env_var_prefix

    def read_env(self):
        self.temperature = read_env_float(self._prefix + "LLM_TEMPERATURE", required=False, default=0.0)
        self.max_tokens = read_env_int(self._prefix + "LLM_MAX_TOKENS", required=False)
        self.top_p = read_env_float(self._prefix + "LLM_TOP_P", required=False)
        self.n = read_env_int(self._prefix + "LLM_N", required=False)
        self.presence_penalty = read_env_float(self._prefix + "LLM_PRESENCE_PENALTY", required=False)
        self.frequency_penalty = read_env_float(self._prefix + "LLM_FREQUENCY_PENALTY", required=False)

    def build_default_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.temperature.is_some():
            payload.setdefault("temperature", self.temperature.unwrap())
        if self.max_tokens.is_some():
            payload.setdefault("max_tokens", self.max_tokens.unwrap())
        if self.top_p.is_some():
            payload.setdefault("top_p", self.top_p.unwrap())
        if self.n.is_some():
            payload.setdefault("n", self.n.unwrap())
        if self.presence_penalty.is_some():
            payload.setdefault("presence_penalty", self.presence_penalty.unwrap())
        if self.frequency_penalty.is_some():
            payload.setdefault("frequency_penalty", self.frequency_penalty.unwrap())
        return payload
