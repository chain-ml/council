from __future__ import annotations

from typing import Any, Dict, Final, Mapping, Tuple, Type

from council.utils import (
    Parameter,
    must_read_env_str,
    not_empty_validator,
    penalty_validator,
    positive_validator,
    zero_to_one_validator,
    zero_to_two_validator,
)

from ...llm_base import LLMConfigurationBase
from ...llm_config_object import LLMConfigSpec, LLMProviders

_env_var_prefix: Final[str] = "GROQ_"


class GroqLLMConfiguration(LLMConfigurationBase):
    def __init__(self, model: str, api_key: str) -> None:
        """
        Initialize a new instance

        Args:
            api_key (str): the api key
            model (str): the model name
        """
        super().__init__()
        self._model = Parameter.string(name="model", required=True, value=model, validator=not_empty_validator)
        self._api_key = Parameter.string(name="api_key", required=True, value=api_key, validator=not_empty_validator)

        # https://console.groq.com/docs/api-reference#chat
        self._frequency_penalty = Parameter.float(name="frequency_penalty", required=False, validator=penalty_validator)
        self._max_tokens = Parameter.int(name="max_tokens", required=False, validator=positive_validator)
        self._presence_penalty = Parameter.float(name="presence_penalty", required=False, validator=penalty_validator)
        self._seed = Parameter.int(name="seed", required=False)
        self._stop = Parameter.string(name="stop", required=False)
        self._temperature = Parameter.float(
            name="temperature", required=False, default=0.0, validator=zero_to_two_validator
        )
        self._top_p = Parameter.float(name="top_p", required=False, validator=zero_to_one_validator)

    def model_name(self) -> str:
        return self._model.unwrap()

    @property
    def model(self) -> Parameter[str]:
        """
        Groq model name
        """
        return self._model

    @property
    def api_key(self) -> Parameter[str]:
        """
        Groq API Key
        """
        return self._api_key

    @property
    def frequency_penalty(self) -> Parameter[float]:
        """
        Number between -2.0 and 2.0.
        Positive values penalize new tokens based on their existing frequency in the text so far,
        decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self._frequency_penalty

    @property
    def max_tokens(self) -> Parameter[int]:
        """Maximum number of tokens to generate."""
        return self._max_tokens

    @property
    def presence_penalty(self) -> Parameter[float]:
        """
        Number between -2.0 and 2.0.
        Positive values penalize new tokens based on whether they appear in the text so far,
        increasing the model's likelihood to talk about new topics.
        """
        return self._presence_penalty

    @property
    def seed(self) -> Parameter[int]:
        """Random seed for generation."""
        return self._seed

    @property
    def stop(self) -> Parameter[str]:
        """Stop sequence."""
        return self._stop

    @property
    def temperature(self) -> Parameter[float]:
        """
        What sampling temperature to use, between 0 and 2.
        """
        return self._temperature

    @property
    def top_p(self) -> Parameter[float]:
        """Nucleus sampling threshold."""
        return self._top_p

    def params_to_args(self) -> Dict[str, Any]:
        """Convert parameters to options dict"""
        return {
            "frequency_penalty": self.frequency_penalty.value,
            "max_tokens": self.max_tokens.value,
            "presence_penalty": self.presence_penalty.value,
            "seed": self.seed.value,
            "stop": self.stop.value,
            "temperature": self.temperature.value,
            "top_p": self.top_p.value,
        }

    @staticmethod
    def from_env() -> GroqLLMConfiguration:
        api_key = must_read_env_str(_env_var_prefix + "API_KEY")
        model = must_read_env_str(_env_var_prefix + "LLM_MODEL")
        config = GroqLLMConfiguration(model=model, api_key=api_key)
        return config

    @classmethod
    def from_spec(cls, spec: LLMConfigSpec) -> GroqLLMConfiguration:
        spec.check_provider(LLMProviders.Groq)

        api_key = spec.provider.must_get_value("apiKey")
        model = spec.provider.must_get_value("model")
        config = GroqLLMConfiguration(model=str(model), api_key=str(api_key))

        if spec.parameters is not None:
            param_mapping: Mapping[str, Tuple[Parameter, Type]] = {
                "frequencyPenalty": (config.frequency_penalty, float),
                "maxTokens": (config.max_tokens, int),
                "presencePenalty": (config.presence_penalty, float),
                "seed": (config.seed, int),
                "stop": (config.stop, list),
                "temperature": (config.temperature, float),
                "topP": (config.top_p, float),
            }

            for key, (param, type_conv) in param_mapping.items():
                value = spec.parameters.get(key)
                if value is not None:
                    param.set(type_conv(value))

        return config
